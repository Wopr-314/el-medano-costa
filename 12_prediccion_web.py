# -*- coding: utf-8 -*-
"""
12_prediccion_web.py

Genera la prediccion que publica la web. Pensado para ejecutarse SOLO (sin
Spyder) desde una GitHub Action cada hora, y tambien a mano para probar.

    python 12_prediccion_web.py --salida web

NO necesita OpenDrift, ni PyTorch, ni el entorno completo: solo numpy,
requests, onnxruntime, matplotlib y copernicusmarine. Eso hace que la Action
instale en menos de un minuto en vez de varios.

Lee de la carpeta del modelo (generada por 11_exportar_onnx.py):
    emulador.onnx · estaticos.bin · metadatos.json
Escribe en la carpeta de salida:
    pluma.png       overlay transparente, georreferenciado por metadatos
    prediccion.json condiciones, exposicion por playa, avisos y sello de tiempo

Las credenciales de CMEMS se leen de las variables de entorno
CMEMS_USUARIO y CMEMS_PASSWORD. En la Action van como *secrets* del
repositorio: NUNCA deben acabar en el HTML ni en el codigo.
"""

import argparse
import datetime as dt
import json
import os
import sys
import numpy as np

DUR_H = 5                    # misma ventana con la que se entreno
PLAYAS = {
    "El Médano centro": (28.0435, -16.5382),
    "El Cabezo":        (28.0457, -16.5333),
    "Leocadio Machado": (28.0378, -16.5425),
    "La Tejita":        (28.0313, -16.5536),
}
UMBRAL_REL = 0.06
ALFA_MIN, ALFA_MAX = 0.35, 0.95


def log(*a):
    print(*a, flush=True)


def cargar_modelo(dir_modelo):
    import onnxruntime as ort
    meta = json.loads((dir_modelo / "metadatos.json").read_text(encoding="utf-8"))
    g = meta["rejilla"]
    nx, ny = g["nx"], g["ny"]
    est = np.fromfile(dir_modelo / "estaticos.bin", dtype=np.float32).reshape(2, nx, ny)
    ses = ort.InferenceSession(str(dir_modelo / "emulador.onnx"),
                               providers=["CPUExecutionProvider"])
    log(f"Modelo cargado · rejilla {nx}x{ny} · {meta['modelo']}")
    return ses, est, meta


def viento(lat, lon, ahora):
    """Viento medio de la ventana, en componentes CF (hacia donde sopla)."""
    import requests
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat:.4f}&longitude={lon:.4f}"
           "&hourly=wind_speed_10m,wind_direction_10m"
           "&wind_speed_unit=ms&past_days=1&forecast_days=2&timezone=UTC")
    h = requests.get(url, timeout=45).json()["hourly"]
    t = np.array([dt.datetime.fromisoformat(x) for x in h["time"]])
    m = (t >= ahora) & (t <= ahora + dt.timedelta(hours=DUR_H))
    if m.sum() == 0:
        raise RuntimeError("Open-Meteo no cubre la ventana pedida")
    v = np.array(h["wind_speed_10m"], float)[m]
    d = np.array(h["wind_direction_10m"], float)[m]
    r = np.radians(d + 180)
    return float((v*np.sin(r)).mean()), float((v*np.cos(r)).mean()), float(v.mean()), float(d.mean())


def corriente(bbox, ahora, dataset_id):
    """Corriente y altura del mar de CMEMS, muestreadas sobre el dominio."""
    import copernicusmarine as cm
    usuario = os.environ.get("CMEMS_USUARIO")
    clave = os.environ.get("CMEMS_PASSWORD")
    if not usuario or not clave:
        raise RuntimeError("Faltan CMEMS_USUARIO / CMEMS_PASSWORD en el entorno")
    ds = cm.open_dataset(dataset_id=dataset_id, username=usuario, password=clave)

    def primera(*c):
        for x in c:
            if x in ds.variables:
                return x
        return None
    un = primera("eastward_sea_water_velocity", "uo")
    vn = primera("northward_sea_water_velocity", "vo")
    sn = primera("sea_surface_height", "zos")
    if un is None or vn is None:
        raise RuntimeError(f"No encuentro las corrientes. Hay: {list(ds.variables)}")

    sel = ds[[x for x in (un, vn, sn) if x]].sel(
        latitude=slice(bbox[1], bbox[3]), longitude=slice(bbox[0], bbox[2]),
        time=slice(np.datetime64(ahora), np.datetime64(ahora + dt.timedelta(hours=DUR_H)))
    ).compute()
    U, V = np.asarray(sel[un]), np.asarray(sel[vn])
    if U.size == 0:
        raise RuntimeError("CMEMS no devolvió datos para esa ventana")
    ssh = float(np.nanmean(np.asarray(sel[sn]))) if sn else 0.0
    return (float(np.nanmean(U)), float(np.nanmean(V)),
            float(np.nanstd(U)), float(np.nanstd(V)), ssh)


def predecir(ses, est, meta, x7):
    n = meta["normalizacion"]
    mu = np.array(n["media"], np.float32)
    sd = np.array(n["desviacion"], np.float32)
    sd[sd < 1e-8] = 1.0
    z = ((np.array(x7, np.float32) - mu) / sd).astype(np.float32)
    nx, ny = meta["rejilla"]["nx"], meta["rejilla"]["ny"]
    cond = np.broadcast_to(z.reshape(1, -1, 1, 1), (1, len(z), nx, ny))
    entrada = np.concatenate([cond, est[None]], axis=1).astype(np.float32)
    salida = ses.run(None, {"entrada": entrada})[0][0, 0]
    campo = np.expm1(salida * n["escala_salida"])
    campo[est[0] > 0.5] = 0.0          # nada de densidad sobre tierra
    return campo, z


def pintar(campo, ruta):
    """Overlay PNG con transparencia. Paleta recortada de turbo: empieza en
    turquesa porque sobre el mar oscuro el azul inicial se pierde."""
    from matplotlib import cm as mcm
    from matplotlib.colors import PowerNorm
    from PIL import Image
    vmax = float(campo.max())
    umbral = max(vmax * UMBRAL_REL, 1e-6)
    paleta = mcm.get_cmap("turbo") if hasattr(mcm, "get_cmap") else None
    if paleta is None:
        import matplotlib.pyplot as plt
        paleta = plt.get_cmap("turbo")
    norm = PowerNorm(0.5, vmin=umbral, vmax=max(vmax, umbral*2))
    # imagen en orientacion de mapa: filas = latitud descendente
    c = campo.T[::-1]
    rgba = paleta(0.32 + 0.68*np.clip(norm(c), 0, 1))
    frac = np.clip((c - umbral)/max(vmax-umbral, 1e-9), 0, 1)**0.4
    rgba[..., 3] = np.where(c >= umbral, ALFA_MIN + (ALFA_MAX-ALFA_MIN)*frac, 0.0)
    Image.fromarray((rgba*255).astype(np.uint8), "RGBA").save(ruta)
    return vmax, umbral


def exposicion(campo, meta, radio_m=500):
    g = meta["rejilla"]
    r = int(round(radio_m / g["celda_m"]))
    out = {}
    for nombre, (la, lo) in PLAYAS.items():
        i = int(round((lo - g["lon_min"]) / g["delta_lon"]))
        j = int(round((la - g["lat_min"]) / g["delta_lat"]))
        sub = campo[max(0, i-r):i+r+1, max(0, j-r):j+r+1]
        out[nombre] = float(sub.sum()) if sub.size else 0.0
    mx = max(max(out.values()), 1e-9)
    return {k: round(100*v/mx) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="web", help="carpeta con emulador.onnx")
    p.add_argument("--salida", default="web", help="carpeta donde publicar")
    p.add_argument("--dataset", default="cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m")
    a = p.parse_args()

    from pathlib import Path
    dm, ds_ = Path(a.modelo), Path(a.salida)
    ds_.mkdir(parents=True, exist_ok=True)

    ses, est, meta = cargar_modelo(dm)
    g = meta["rejilla"]
    lon_max = g["lon_min"] + g["nx"]*g["delta_lon"]
    lat_max = g["lat_min"] + g["ny"]*g["delta_lat"]
    latc, lonc = (g["lat_min"]+lat_max)/2, (g["lon_min"]+lon_max)/2

    ahora = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    log(f"Instante: {ahora:%Y-%m-%d %H:%M} UTC (+{DUR_H} h)")

    vu, vv, vmod, vdir = viento(latc, lonc, ahora)
    log(f"Viento {vmod:.1f} m/s desde {vdir:.0f}°")
    cu, cv, cus, cvs, ssh = corriente(
        (g["lon_min"], g["lat_min"], lon_max, lat_max), ahora, a.dataset)
    log(f"Corriente u={cu:+.3f} v={cv:+.3f} · SSH {ssh:+.3f}")

    x7 = [vu, vv, cu, cv, cus, cvs, ssh]
    campo, z = predecir(ses, est, meta, x7)
    vmax, umbral = pintar(campo, ds_ / "pluma.png")
    log(f"Campo máx {vmax:.1f} · umbral {umbral:.2f}")

    # Fuera del rango de entrenamiento la prediccion es extrapolacion: hay que
    # decirlo en la propia web, no esconderlo.
    nombres = meta["normalizacion"]["nombres"]
    fuera = [n for n, zz in zip(nombres, z) if abs(zz) > 3]

    exp = exposicion(campo, meta)
    salida = {
        "generado_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "instante_utc": ahora.isoformat(),
        "horas": DUR_H,
        "modelo": meta["modelo"],
        "condiciones": dict(zip(nombres, [round(v, 4) for v in x7])),
        "viento_resumen": {"velocidad_ms": round(vmod, 1), "desde_grados": round(vdir)},
        "fuera_de_rango": fuera,
        "exposicion_playas": exp,
        "menor_exposicion": min(exp, key=exp.get),
        "limites": {"lon_min": g["lon_min"], "lat_min": g["lat_min"],
                    "lon_max": lon_max, "lat_max": lat_max},
        "precision": meta["precision"],
        "avisos": meta["avisos"],
        "focos": meta["focos"],
    }
    (ds_ / "prediccion.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Publicado en {ds_}: pluma.png + prediccion.json")
    if fuera:
        log(f"AVISO: fuera del rango de entrenamiento: {', '.join(fuera)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
