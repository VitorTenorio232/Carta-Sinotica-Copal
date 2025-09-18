# -*- coding: utf-8 -*-
"""
Script Unificado e Automático - GFS + METAR + Satélite GOES-19
Versão 11.1 - Versão Final.

Funcionalidades:
 - Plota cartas sinóticas com dados GFS e METAR.
 - Plota imagens de satélite (estáticas ou GIFs) com zoom regional para o Brasil.
 - Aba "Histórico" totalmente funcional para GFS e imagens estáticas de satélite.
 - Exibe as divisas dos estados do Brasil (fonte IBGE 2024) em todos os mapas.
 - Interface gráfica unificada e adaptativa.
 - Limpeza automática de todos os dados de GFS e METAR ao final da execução.
 - Ponto na Cidade de Itajubá/MG em todos os produtos
 
Criado por: VitorTenorio (COPAL)
"""
import os
import sys
import glob
import warnings
from datetime import datetime, timedelta, timezone
import urllib.request
import requests
import threading
import customtkinter as ctk
from tkinter import filedialog
import subprocess
import colorsys
import traceback
import io
import zipfile
import imageio.v2 as imageio
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

# Import opcionais
try:
    from metpy.plots import StationPlot, sky_cover, current_weather
    from metpy.io import metar
    from metpy.calc import wind_components
    from metpy.units import units
except ImportError:
    StationPlot, sky_cover, current_weather, metar, wind_components, units = None, None, None, None, None, None

# =============================================================================
# LÓGICA DE BACKEND
# =============================================================================

REGIONS = {
    "South_America": [-120, 0, -65, 15], "North_America": [-170, -50, 10, 75],
    "Europe": [-30, 40, 35, 72], "Africa": [-20, 55, -35, 38], "Asia": [40, 150, -10, 60],
    "Oceania": [110, 180, -50, 0], "World": [-180, 180, -90, 90]
}
ZOOM_REGIONS_BR = {
    "Brasil": [-75, -34, -34, 6],
    "Sudeste": [-52, -38, -26, -18],
    "Sul": [-58, -48, -34, -22],
    "Nordeste": [-49, -34, -19, -1],
    "Norte/CO": [-75, -45, -17, 6]
}

# --- Funções de Download ---
def find_latest_gfs_run(log_function):
    now_utc = datetime.now(timezone.utc)
    log_function("Procurando a rodada mais recente do GFS disponível...")
    for hours_back in range(48):
        check_time = now_utc - timedelta(hours=hours_back)
        date_str = check_time.strftime('%Y%m%d')
        for run in ['18', '12', '06', '00']:
            if check_time.hour >= int(run):
                url_check = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date_str}/{run}/atmos/gfs.t{run}z.pgrb2.0p25.f000.idx"
                try:
                    response = requests.head(url_check, timeout=5)
                    if response.status_code == 200:
                        log_function(f"✅ Rodada mais recente encontrada: {date_str} {run}Z")
                        return date_str, run
                except requests.exceptions.RequestException:
                    continue
    return None, None

def download_gfs_data(date, run_hour, extent, gfs_save_dir, log_function, progress_callback=None):
    min_lon, max_lon, min_lat, max_lat = [str(val) for val in extent]
    gfs_vars = ['HGT', 'PRMSL', 'MSLET', 'TMP', 'RH', 'CAPE']
    var_str = "".join([f"&var_{v}=on" for v in gfs_vars])
    url = (f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'
           f'file=gfs.t{run_hour}z.pgrb2.0p25.f000&all_lev=on{var_str}&subregion='
           f'&leftlon={min_lon}&rightlon={max_lon}&toplat={max_lat}&bottomlat={min_lat}'
           f'&dir=%2Fgfs.{date}%2F{run_hour}%2Fatmos')
    file_name = f'gfs.t{run_hour}z.pgrb2.0p25.f000.{date}.grib2'
    file_path = os.path.join(gfs_save_dir, file_name)
    if os.path.exists(file_path):
        log_function(f"Usando arquivo GFS existente: {file_name}")
        if progress_callback: progress_callback(1.0)
        return file_path
    log_function(f"\nBaixando arquivo GFS: {file_name}")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0 and progress_callback:
                    progress = downloaded_size / total_size
                    progress_callback(progress)
        return file_path
    except requests.exceptions.RequestException as e:
        log_function(f"⚠️ Erro ao baixar o arquivo GFS: {e}")
        return None

def download_satellite_data(config, target_time, save_dir, log_function, silent=False):
    ftp_base = 'http://ftp.cptec.inpe.br/goes/goes19/retangular/'
    file_url = f"{ftp_base}ch{config['ch']}/{target_time.strftime('%Y/%m')}/{config['prefix']}_{target_time.strftime('%Y%m%d%H%M')}.nc"
    file_name = os.path.basename(file_url)
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        if not silent: log_function(f"Usando arquivo existente: {file_name}")
        return file_path, target_time
    try:
        response = requests.get(file_url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        with open(file_path, 'wb') as f: f.write(response.content)
        if not silent: log_function(f"✅ Download concluído: {file_name}")
        return file_path, target_time
    except requests.exceptions.RequestException:
        return None, None

def download_cpt_files(save_dir, log_function):
    cpt_files = {"ir.cpt": "https://raw.githubusercontent.com/VitorTenorio232/Carta-Sinotica-Copal/main/ir.cpt",
                 "wv.cpt": "https://raw.githubusercontent.com/VitorTenorio232/Carta-Sinotica-Copal/main/wv.cpt"}
    os.makedirs(save_dir, exist_ok=True)
    for name, url in cpt_files.items():
        if not os.path.exists(os.path.join(save_dir, name)):
            try:
                log_function(f"Baixando paleta de cores: {name}...")
                response = requests.get(url, timeout=30); response.raise_for_status()
                with open(os.path.join(save_dir, name), 'wb') as f: f.write(response.content)
            except requests.exceptions.RequestException as e: log_function(f"⚠️ Erro ao baixar {name}: {e}")

def download_brazil_shapefiles(save_dir, log_function):
    shapefile_url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/Brasil/BR_UF_2024.zip"
    shapefile_name = "BR_UF_2024.shp"
    os.makedirs(save_dir, exist_ok=True)
    shp_path = os.path.join(save_dir, shapefile_name)
    if os.path.exists(shp_path): return shp_path
    log_function(f"\nBaixando shapefile dos estados do Brasil (versão 2024)...")
    try:
        response = requests.get(shapefile_url, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z: z.extractall(save_dir)
        log_function("✅ Shapefile baixado e extraído com sucesso.")
        for extracted_file in os.listdir(save_dir):
            if extracted_file.endswith('.shp'):
                if extracted_file.upper() != shapefile_name.upper():
                    os.rename(os.path.join(save_dir, extracted_file), shp_path)
                return shp_path
    except Exception as e:
        log_function(f"⚠️ Erro ao baixar/extrair shapefile: {e}")
        return None

# --- Funções Auxiliares ---
def loadCPT(path, log_function):
    try:
        with open(path) as f: lines = f.readlines()
    except Exception as e:
        log_function(f"⚠️ Erro ao ler paleta de cores {path}: {e}"); return None
    x, r, g, b, xt, rt, gt, bt = [], [], [], [], 0, 0, 0, 0
    for l in lines:
        if l.startswith('#') or l.strip() and l[0] in ['B', 'F', 'N']: continue
        try:
            vals = l.split()
            if len(vals) >= 8:
                x.append(float(vals[0])); r.append(float(vals[1])); g.append(float(vals[2])); b.append(float(vals[3]))
                xt, rt, gt, bt = float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])
        except (ValueError, IndexError): continue
    x.append(xt); r.append(rt); g.append(gt); b.append(bt)
    x, r, g, b = np.array(x), np.array(r), np.array(g), np.array(b)
    x_norm = (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) else 0
    red   = [(x_norm[i], r[i]/255., r[i]/255.) for i in range(len(x))]
    green = [(x_norm[i], g[i]/255., g[i]/255.) for i in range(len(x))]
    blue  = [(x_norm[i], b[i]/255., b[i]/255.) for i in range(len(x))]
    return {'red': red, 'green': green, 'blue': blue}

def find_pressure_var(ds):
    candidates = ['prmsl', 'msl', 'mslet', 'air_pressure_at_sea_level', 'PMSL']
    for c in candidates:
        if c in ds.data_vars: return c
    return None

def get_temperature_data_array(gfs_file_path):
    filters_to_try = [({'shortName': '2t'}, 't2m'), ({'shortName': 't2m'}, 't2m'), ({'shortName': 'tmp', 'typeOfLevel': 'surface'}, 'tmp')]
    for filt, var_name in filters_to_try:
        try:
            ds = xr.open_dataset(gfs_file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': filt})
            if var_name in ds: return ds[var_name]
        except Exception: continue
    raise ValueError("Variável de temperatura não encontrada.")

def make_2d_grid_from_da(da):
    lon_name, lat_name = None, None
    for name in ['longitude', 'lon']:
        if name in da.coords: lon_name = name; break
    for name in ['latitude', 'lat']:
        if name in da.coords: lat_name = name; break
    if not lon_name or not lat_name: raise ValueError("Coordenadas lon/lat não detectadas.")
    lons, lats = da[lon_name].values, da[lat_name].values
    lons = np.where(lons > 180, lons - 360, lons)
    lon_sort_indices, lat_sort_indices = np.argsort(lons), np.argsort(lats)
    lons, lats = lons[lon_sort_indices], lats[lat_sort_indices]
    da = da.isel({lon_name: lon_sort_indices, lat_name: lat_sort_indices})
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d, da.values
    
def open_file(filepath):
    try:
        if sys.platform == "win32": os.startfile(filepath)
        else: subprocess.run(["open", filepath], check=True, timeout=10)
    except Exception as e: print(f"Erro ao abrir arquivo: {e}")

def cleanup_data_folders(base_dir, log_function):
    log_function("\nIniciando limpeza de arquivos de dados...")
    dirs_to_clean = [os.path.join(base_dir, "gfs_data"), os.path.join(base_dir, "metar_data")]
    for data_dir in dirs_to_clean:
        if not os.path.isdir(data_dir): continue
        log_function(f"Limpando pasta: {os.path.basename(data_dir)}...")
        files = glob.glob(os.path.join(data_dir, "*"))
        if not files:
            log_function(" - Pasta já está vazia.")
            continue
        for f in files:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError as e:
                log_function(f" - Erro ao remover {os.path.basename(f)}: {e}")
    log_function("Limpeza de dados concluída.")
    
def nearest_synoptic_hour():
    dt_utc = datetime.now(timezone.utc)
    h = max((hh for hh in [0, 6, 12, 18] if hh <= dt_utc.hour), default=-1)
    if h == -1: dt_utc -= timedelta(days=1); h = 18
    return dt_utc.replace(hour=h, minute=0, second=0, microsecond=0)

# --- Funções de Plotagem ---
def plot_gfs_map(region, data_dirs, zoom_key, gfs_file_path, log_function, historical_datetime=None, progress_callback=None, produto_adicional="Carta Padrão"):
    map_save_dir = data_dirs['maps']
    metar_save_dir = data_dirs['metar_data']
    
    fig = plt.figure(figsize=(15, 12)); ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    current_extent = REGIONS[region]
    if region == "South_America" and zoom_key in ZOOM_REGIONS_BR:
        log_function(f"Aplicando zoom para '{zoom_key}'.")
        current_extent = ZOOM_REGIONS_BR[zoom_key]
    
    ax.set_extent(current_extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.6, edgecolor='black', zorder=2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), color='#E0D5B1', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), color='#B0C4DE', zorder=0)

    try:
        shapefile = os.path.join(data_dirs['shapefiles'], 'BR_UF_2024.shp')
        if os.path.exists(shapefile):
            ax.add_geometries(shpreader.Reader(shapefile).geometries(), ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.5, zorder=2)
    except Exception as e: log_function(f"Aviso: Não foi possível plotar os estados do Brasil. {e}")

    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    valid_time_str = ""
    try:
        temp_da_for_time = get_temperature_data_array(gfs_file_path)
        valid_time = pd.to_datetime(temp_da_for_time['valid_time'].values)
        valid_time_str = f"GFS {valid_time.strftime('%Y-%m-%d %HZ')}"
    except Exception: pass
    if progress_callback: progress_callback(0.55)

    try:
        if "Carta Padrão" in produto_adicional:
            ds_msl = xr.open_dataset(gfs_file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            press_var = find_pressure_var(ds_msl); da_press = ds_msl[press_var] / 100.0
            
            lon2d_full, lat2d_full, press_vals_full = make_2d_grid_from_da(da_press)
            cs = ax.contour(lon2d_full, lat2d_full, press_vals_full, levels=np.arange(960, 1061, 4), colors='black', linewidths=0.7, transform=ccrs.PlateCarree(), zorder=3)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
            
            ds_iso = xr.open_dataset(gfs_file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'}})
            thickness = ds_iso['gh'].sel(isobaricInhPa=500) - ds_iso['gh'].sel(isobaricInhPa=1000)
            lon2d_t, lat2d_t, thickness_vals = make_2d_grid_from_da(thickness)
            cs_b = ax.contour(lon2d_t, lat2d_t, thickness_vals, levels=np.arange(4800, 5400, 60), colors='blue', linestyles='--', linewidths=0.8, transform=ccrs.PlateCarree(), zorder=3)
            ax.clabel(cs_b, inline=True, fontsize=7, fmt='%1.0f')
            cs_r = ax.contour(lon2d_t, lat2d_t, thickness_vals, levels=np.arange(5400, 6001, 60), colors='red', linestyles='--', linewidths=0.8, transform=ccrs.PlateCarree(), zorder=3)
            ax.clabel(cs_r, inline=True, fontsize=7, fmt='%1.0f')

        elif "CAPE" in produto_adicional:
            ds_cape = xr.open_dataset(gfs_file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'cape', 'typeOfLevel': 'surface'}})
            lon2d_c, lat2d_c, cape_vals = make_2d_grid_from_da(ds_cape['cape'])
            cf = ax.contourf(lon2d_c, lat2d_c, cape_vals, levels=[250, 500, 1000, 1500, 2000, 2500, 3000, 4000], cmap='plasma', extend='max', alpha=0.7, transform=ccrs.PlateCarree())
            fig.colorbar(cf, ax=ax, label='CAPE (J kg⁻¹)', orientation='vertical', pad=0.02, shrink=0.8)
            
        elif "Temperatura" in produto_adicional:
            temp_da = get_temperature_data_array(gfs_file_path)
            lon2d_temp, lat2d_temp, temp_vals = make_2d_grid_from_da(temp_da)
            cf = ax.contourf(lon2d_temp, lat2d_temp, temp_vals - 273.15, levels=np.arange(-20, 31, 5), cmap='coolwarm', extend='both', alpha=0.8, transform=ccrs.PlateCarree())
            fig.colorbar(cf, ax=ax, label='Temperatura de Superfície (°C)', orientation='vertical', pad=0.02, shrink=0.8)
            
        elif "Umidade" in produto_adicional:
            ds_rh = xr.open_dataset(gfs_file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'r', 'typeOfLevel': 'isobaricInhPa'}})
            lon2d_rh, lat2d_rh, rh_vals = make_2d_grid_from_da(ds_rh['r'].sel(isobaricInhPa=700))
            cf = ax.contourf(lon2d_rh, lat2d_rh, rh_vals, levels=np.arange(70, 101, 5), cmap='Greens', extend='min', alpha=0.6, transform=ccrs.PlateCarree())
            fig.colorbar(cf, ax=ax, label='Umidade Relativa (%) em 700 hPa', orientation='vertical', pad=0.02, shrink=0.8)

        if progress_callback: progress_callback(0.75)
    except Exception as e:
        log_function(f"⚠️ Erro ao processar produto GFS: {e}\n{traceback.format_exc()}")
    
    if "Carta Padrão" in produto_adicional and metar:
        try:
            rounded_time = historical_datetime if historical_datetime else nearest_synoptic_hour()
            metar_time_str = rounded_time.strftime('%Y%m%d_%H00')
            metar_url = f"https://thredds.ucar.edu/thredds/fileServer/noaaport/text/metar/metar_{metar_time_str}.txt"
            local_metar = os.path.join(metar_save_dir, f"metar_{metar_time_str}.txt")
            if not os.path.exists(local_metar):
                log_function(f"Baixando METAR para {rounded_time.strftime('%HZ')}...")
                urllib.request.urlretrieve(metar_url, local_metar)
            
            data = metar.parse_metar_file(local_metar).dropna(subset=['longitude', 'latitude', 'air_temperature', 'dew_point_temperature', 'air_pressure_at_sea_level'])
            
            lon_min, lon_max, lat_min, lat_max = current_extent
            data = data[(data['longitude'] >= lon_min) & (data['longitude'] <= lon_max) & (data['latitude'] >= lat_min) & (data['latitude'] <= lat_max)].copy()

            if data.empty:
                log_function("Aviso: Nenhum dado METAR na área do mapa.")
            else:
                u, v = wind_components(data['wind_speed'].values * units.knots, data['wind_direction'].values * units.deg)
                stationplot = StationPlot(ax, data['longitude'].values, data['latitude'].values, clip_on=True, transform=ccrs.PlateCarree(), fontsize=4)
                try: stationplot.plot_parameter('NW', data['air_temperature'], color='red')
                except Exception: pass
                try: stationplot.plot_parameter('SW', data['dew_point_temperature'], color='darkgreen')
                except Exception: pass
                try: stationplot.plot_parameter('NE', data['air_pressure_at_sea_level'], formatter=lambda v: format(10 * v, '.0f')[-3:])
                except Exception: pass
                try: stationplot.plot_symbol('C', data['cloud_coverage'], sky_cover)
                except Exception: pass
                try: stationplot.plot_barb(u, v)
                except Exception: pass
                valid_time_str += f" | METAR {rounded_time.strftime('%HZ')}"
        except Exception as e: log_function(f"⚠️ Erro ao processar METAR: {e}")

    # Adiciona a estrela roxa em Itajubá
    itajuba_lon, itajuba_lat = -45.45, -22.42
    ax.plot(itajuba_lon, itajuba_lat, marker='*', color='purple', markersize=12, transform=ccrs.PlateCarree(), zorder=10)
    
    # Adiciona os títulos
    plt.title(f"{produto_adicional} - {region}\n{valid_time_str}", fontsize=14, loc='left')
    plt.title("Criado por: VitorTenorio (COPAL)", fontsize=10, loc='right')
    
    fig.tight_layout(pad=3.0)
    fname = os.path.join(map_save_dir, f"map_{region}_{produto_adicional.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    log_function(f"\n✅ Figura salva em: {fname}")
    if progress_callback: progress_callback(0.95)
    plt.close(fig)
    return fname

def plot_satellite_map(map_save_dir, data_dirs, zoom_key, log_function, historical_datetime=None, progress_callback=None, produto_adicional="Satélite - IR", is_historical=False):
    fig = plt.figure(figsize=(15, 12)); ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    try:
        sat_config = { "Infravermelho (IR)": {"ch": "13", "prefix": "S10165545", "cmap": "ir.cpt", "vmin": -103.0, "vmax": 84.0},
                       "Visível (VIS)":      {"ch": "02", "prefix": "S10165534", "cmap": "Greys_r", "vmin": 0, "vmax": 100},
                       "Vapor d'Água (WV)":  {"ch": "08", "prefix": "S10165540", "cmap": "wv.cpt", "vmin": -112.0, "vmax": 77.0} }
        key = [k for k in sat_config if k in produto_adicional][0]
        config = sat_config[key]
        base_target_time = historical_datetime if is_historical else datetime.now(timezone.utc)
        
        file_path, sat_time = None, None
        for i in range(6):
            target_time = base_target_time - timedelta(minutes=10 * i)
            rounded_minute = int(target_time.minute / 10) * 10; search_time = target_time.replace(minute=rounded_minute, second=0, microsecond=0)
            file_path, sat_time = download_satellite_data(config, search_time, data_dirs['satellite'], log_function, silent=True)
            if file_path:
                log_function(f"Encontrada imagem de satélite para {sat_time.strftime('%Y-%m-%d %H:%M')} UTC.")
                break
            if is_historical: break
        
        if not file_path: log_function("Nenhuma imagem de satélite encontrada."); plt.close(fig); return None
        if progress_callback: progress_callback(0.60)
        
        ds = xr.open_dataset(file_path); data = ds['Band1'].values / 100.0
        if "Visível" not in key: data -= 273.15
        
        lat_min, lat_max, lon_min, lon_max = ds['lat'].min().item(), ds['lat'].max().item(), ds['lon'].min().item(), ds['lon'].max().item()
        base_extent = [lon_min, lon_max, lat_min, lat_max]
        
        if zoom_key in ZOOM_REGIONS_BR:
            ax.set_extent(ZOOM_REGIONS_BR[zoom_key], crs=ccrs.PlateCarree())
        else:
            ax.set_extent(base_extent, crs=ccrs.PlateCarree())
        
        cmap = config['cmap']
        if isinstance(cmap, str) and cmap.endswith('.cpt'):
            cmap_dict = loadCPT(os.path.join(data_dirs['cpt'], cmap), log_function)
            if cmap_dict: cmap = LinearSegmentedColormap('cpt_conv', cmap_dict)
        
        ax.imshow(np.flipud(data), origin='upper', extent=base_extent, cmap=cmap, vmin=config['vmin'], vmax=config['vmax'], transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.6, edgecolor='black', zorder=2)
        try:
            shapefile = os.path.join(data_dirs['shapefiles'], 'BR_UF_2024.shp')
            if os.path.exists(shapefile):
                ax.add_geometries(shpreader.Reader(shapefile).geometries(), ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.5, zorder=2)
        except Exception as e: log_function(f"Aviso: Não foi possível plotar os estados do Brasil. {e}")
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Adiciona a estrela roxa em Itajubá
        itajuba_lon, itajuba_lat = -45.45, -22.42
        ax.plot(itajuba_lon, itajuba_lat, marker='*', color='purple', markersize=12, transform=ccrs.PlateCarree(), zorder=10)

        # Adiciona os títulos
        plt.title(f"GOES-19 - {key}\n{sat_time.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=14, loc='left')
        plt.title("Criado por: VitorTenorio (COPAL)", fontsize=10, loc='right')
        
        fig.tight_layout(pad=3.0)
        fname = os.path.join(map_save_dir, f"satellite_{key.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        log_function(f"\n✅ Figura salva em: {fname}")
        if progress_callback: progress_callback(0.95)
        plt.close(fig); return fname
    except Exception as e:
        log_function(f"⚠️ Erro na plotagem do satélite: {e}\n{traceback.format_exc()}"); plt.close(fig); return None

def create_satellite_gif(map_save_dir, data_dirs, zoom_key, log_function, produto_adicional, progress_callback):
    log_function("\nIniciando criação de GIF animado (30 quadros)...")
    
    sat_config = { "Infravermelho (IR)": {"ch": "13", "prefix": "S10165545", "cmap": "ir.cpt", "vmin": -103.0, "vmax": 84.0},
                  "Visível (VIS)":      {"ch": "02", "prefix": "S10165534", "cmap": "Greys_r", "vmin": 0, "vmax": 100},
                  "Vapor d'Água (WV)":  {"ch": "08", "prefix": "S10165540", "cmap": "wv.cpt", "vmin": -112.0, "vmax": 77.0} }
    key = [k for k in sat_config if k in produto_adicional][0]
    config = sat_config[key]
    
    frames_dir = os.path.join(data_dirs['maps'], "temp_frames")
    if os.path.exists(frames_dir): shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    
    frame_paths, base_time = [], datetime.now(timezone.utc)
    
    for i in range(30):
        target_time = base_time - timedelta(minutes=10 * i)
        rounded_minute = int(target_time.minute / 10) * 10
        frame_time = target_time.replace(minute=rounded_minute, second=0, microsecond=0)
        
        log_function(f"Buscando quadro {i+1}/30 para {frame_time.strftime('%H:%M')} UTC...")
        file_path, _ = download_satellite_data(config, frame_time, data_dirs['satellite'], log_function, silent=True)
        
        if file_path:
            frame_fname = plot_satellite_frame(frames_dir, data_dirs, zoom_key, file_path, config, key, frame_time, log_function)
            if frame_fname: frame_paths.append(frame_fname)
        if progress_callback: progress_callback(0.1 + (i/30 * 0.7))

    if not frame_paths:
        log_function("⚠️ Nenhum quadro encontrado para criar a animação.")
        shutil.rmtree(frames_dir)
        return None
        
    log_function("Montando GIF..."); frame_paths.reverse()
    images = [imageio.imread(path) for path in frame_paths]
    gif_fname = os.path.join(map_save_dir, f"satellite_animation_{key.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.gif")
    imageio.mimsave(gif_fname, images, duration=500, loop=0)
    
    shutil.rmtree(frames_dir)
    log_function(f"\n✅ GIF salvo em: {gif_fname}")
    if progress_callback: progress_callback(0.95)
    return gif_fname

def plot_satellite_frame(save_dir, data_dirs, zoom_key, file_path, config, key, timestamp, log_function):
    try:
        fig = plt.figure(figsize=(10, 10)); ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ds = xr.open_dataset(file_path)
        data = ds['Band1'].values / 100.0
        if "Visível" not in key: data -= 273.15
        
        lat_min, lat_max, lon_min, lon_max = ds['lat'].min().item(), ds['lat'].max().item(), ds['lon'].min().item(), ds['lon'].max().item()
        base_extent = [lon_min, lon_max, lat_min, lat_max]
        
        if zoom_key in ZOOM_REGIONS_BR: ax.set_extent(ZOOM_REGIONS_BR[zoom_key], crs=ccrs.PlateCarree())
        else: ax.set_extent(base_extent, crs=ccrs.PlateCarree())
        
        cmap = config['cmap']
        if isinstance(cmap, str) and cmap.endswith('.cpt'):
            cmap_dict = loadCPT(os.path.join(data_dirs['cpt'], cmap), log_function)
            if cmap_dict: cmap = LinearSegmentedColormap('cpt_conv', cmap_dict)
            
        ax.imshow(np.flipud(data), origin='upper', extent=base_extent, cmap=cmap, vmin=config['vmin'], vmax=config['vmax'], transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='white', zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.6, edgecolor='white', zorder=2)
        shapefile = os.path.join(data_dirs['shapefiles'], 'BR_UF_2024.shp')
        if os.path.exists(shapefile):
            ax.add_geometries(shpreader.Reader(shapefile).geometries(), ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=0.5, zorder=2)
        
        # Adiciona a estrela roxa em Itajubá
        itajuba_lon, itajuba_lat = -45.45, -22.42
        ax.plot(itajuba_lon, itajuba_lat, marker='*', color='purple', markersize=12, transform=ccrs.PlateCarree(), zorder=10)
        
        # Adiciona o timestamp e o crédito do criador
        ax.text(0.02, 0.02, timestamp.strftime('%d-%b-%Y %H:%M UTC'), transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
        ax.text(0.98, 0.02, "VitorTenorio (COPAL)", transform=ax.transAxes, color='white', fontsize=10, ha='right', bbox=dict(facecolor='black', alpha=0.5))
        
        fname = os.path.join(save_dir, f"frame_{timestamp.strftime('%Y%m%d%H%M')}.png")
        plt.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig); return fname
    except Exception:
        plt.close(fig); return None
        
# =============================================================================
# GUI
# =============================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Altera o título da janela para incluir o criador
        self.title("Gerador de Cartas e Satélite v11.1 | por VitorTenorio (COPAL)")
        self.geometry("750x850")
        ctk.set_appearance_mode("System"); ctk.set_default_color_theme("blue")
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(3, weight=1)
        self.current_zoom = "Full"

        self.config_frame = ctk.CTkFrame(self)
        self.config_frame.grid(row=0, column=0, padx=10, pady=(10,0), sticky="ew")
        self.config_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.config_frame, text="Pasta de Saída:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        try: script_dir = os.path.dirname(os.path.realpath(__file__))
        except NameError: script_dir = os.getcwd()
        self.path_entry = ctk.CTkEntry(self.config_frame)
        self.path_entry.insert(0, os.path.join(script_dir, "output"))
        self.path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self.config_frame, text="Procurar...", command=self.browse_folder, width=100).grid(row=0, column=2, padx=10, pady=10)
        
        self.tab_view = ctk.CTkTabview(self, anchor="w")
        self.tab_view.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.tab_recente = self.tab_view.add("Mais Recente")
        self.tab_historico = self.tab_view.add("Histórico")
        
        self.produtos_gfs = ["Carta Padrão", "CAPE na Superfície", "Temperatura em 2m", "Umidade Relativa 700hPa"]
        self.produtos_satelite = ["Satélite - Infravermelho (IR)", "Satélite - Visível (VIS)", "Satélite - Vapor d'Água (WV)"]
        
        self.setup_recente_tab()
        self.setup_historico_tab()
        self._update_ui_controls()

        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", mode="determinate")
        self.log_textbox = ctk.CTkTextbox(self, state="disabled", wrap="word")
        self.log_textbox.grid(row=3, column=0, padx=10, pady=(0,10), sticky="nsew")

    def setup_recente_tab(self):
        self.tab_recente.grid_columnconfigure(1, weight=1)
        self.region_label_recente = ctk.CTkLabel(self.tab_recente, text="Região (Download):")
        self.zoom_label_recente = ctk.CTkLabel(self.tab_recente, text="Zoom (Brasil):")
        self.produto_label_recente = ctk.CTkLabel(self.tab_recente, text="Tipo de Produto:")
        self.region_menu_recente = ctk.CTkOptionMenu(self.tab_recente, values=list(REGIONS.keys()), command=self._update_ui_controls)
        self.zoom_button_recente = ctk.CTkSegmentedButton(self.tab_recente, values=["Full", "Brasil", "Sudeste", "Sul", "Nordeste", "Norte/CO"], command=self.on_zoom_select)
        self.produto_menu_recente = ctk.CTkOptionMenu(self.tab_recente, values=self.produtos_gfs + self.produtos_satelite, command=self._update_ui_controls)
        self.gif_checkbox_recente = ctk.CTkCheckBox(self.tab_recente, text="Criar GIF animado (5 horas)")
        self.generate_button_recente = ctk.CTkButton(self.tab_recente, text="Gerar Produto Mais Recente", command=lambda: self.start_task(is_historical=False))
        self.region_label_recente.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.region_menu_recente.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.zoom_label_recente.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.zoom_button_recente.grid(row=1, column=1, padx=10, pady=10, sticky="ew"); self.zoom_button_recente.set("Full")
        self.produto_label_recente.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.produto_menu_recente.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.gif_checkbox_recente.grid(row=3, column=1, padx=10, pady=(0,10), sticky="w")
        self.generate_button_recente.grid(row=4, column=0, columnspan=2, padx=10, pady=(20,10), sticky="ew")

    def setup_historico_tab(self):
        self.tab_historico.grid_columnconfigure(1, weight=1)
        self.produto_label_historico = ctk.CTkLabel(self.tab_historico, text="Tipo de Produto:")
        self.date_label_historico = ctk.CTkLabel(self.tab_historico, text="Data:")
        self.time_label_historico = ctk.CTkLabel(self.tab_historico, text="Horário (UTC):")
        self.region_label_historico = ctk.CTkLabel(self.tab_historico, text="Região (Download):")
        self.zoom_label_historico = ctk.CTkLabel(self.tab_historico, text="Zoom (Brasil):")
        self.produto_menu_historico = ctk.CTkOptionMenu(self.tab_historico, values=self.produtos_gfs + self.produtos_satelite, command=self._update_ui_controls)
        self.date_menu = ctk.CTkOptionMenu(self.tab_historico, values=[(datetime.now(timezone.utc).date() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(11)])
        self.time_menu = ctk.CTkOptionMenu(self.tab_historico, values=[])
        self.region_menu_historico = ctk.CTkOptionMenu(self.tab_historico, values=list(REGIONS.keys()), command=self._update_ui_controls)
        self.zoom_button_historico = ctk.CTkSegmentedButton(self.tab_historico, values=["Full", "Brasil", "Sudeste", "Sul", "Nordeste", "Norte/CO"], command=self.on_zoom_select)
        self.generate_button_historico = ctk.CTkButton(self.tab_historico, text="Gerar Produto Histórico", command=lambda: self.start_task(is_historical=True))
        self.produto_label_historico.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.produto_menu_historico.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.date_label_historico.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.date_menu.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.time_label_historico.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.time_menu.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.region_label_historico.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.region_menu_historico.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
        self.zoom_label_historico.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.zoom_button_historico.grid(row=4, column=1, padx=10, pady=10, sticky="ew"); self.zoom_button_historico.set("Full")
        self.generate_button_historico.grid(row=5, column=0, columnspan=2, padx=10, pady=(20,10), sticky="ew")

    def on_zoom_select(self, selection): self.current_zoom = selection
        
    def _update_ui_controls(self, selection=None):
        produto_recente = self.produto_menu_recente.get()
        region_recente = self.region_menu_recente.get()
        is_sat_recente = "Satélite" in produto_recente
        
        self.gif_checkbox_recente.grid_remove()
        self.region_label_recente.grid_remove(); self.region_menu_recente.grid_remove()
        self.zoom_label_recente.grid_remove(); self.zoom_button_recente.grid_remove()
        
        if is_sat_recente:
            self.gif_checkbox_recente.grid(row=3, column=1, padx=10, pady=(0,10), sticky="w")
            self.zoom_label_recente.grid(row=1, column=0, padx=10, pady=10, sticky="w")
            self.zoom_button_recente.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        else:
            self.region_label_recente.grid(row=0, column=0, padx=10, pady=10, sticky="w")
            self.region_menu_recente.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
            if region_recente == "South_America":
                self.zoom_label_recente.grid(row=1, column=0, padx=10, pady=10, sticky="w")
                self.zoom_button_recente.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
                
        produto_historico = self.produto_menu_historico.get()
        region_historico = self.region_menu_historico.get()
        is_sat_historico = "Satélite" in produto_historico

        self.region_label_historico.grid_remove(); self.region_menu_historico.grid_remove()
        self.zoom_label_historico.grid_remove(); self.zoom_button_historico.grid_remove()
        if is_sat_historico:
            self.zoom_label_historico.grid(row=4, column=0, padx=10, pady=10, sticky="w")
            self.zoom_button_historico.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
        else:
            self.region_label_historico.grid(row=3, column=0, padx=10, pady=10, sticky="w")
            self.region_menu_historico.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
            if region_historico == "South_America":
                self.zoom_label_historico.grid(row=4, column=0, padx=10, pady=10, sticky="w")
                self.zoom_button_historico.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
        
        gfs_times = ['00:00', '06:00', '12:00', '18:00']
        sat_times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 10)]
        new_times = sat_times if is_sat_historico else gfs_times
        current_time = self.time_menu.get()
        self.time_menu.configure(values=new_times)
        if current_time not in new_times: self.time_menu.set(new_times[0])

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path: self.path_entry.delete(0, "end"); self.path_entry.insert(0, folder_path)
    def log(self, message):
        self.log_textbox.configure(state="normal"); self.log_textbox.insert("end", message + "\n"); self.log_textbox.configure(state="disabled"); self.log_textbox.see("end")
    def prepare_ui_for_generation(self):
        self.generate_button_recente.configure(state="disabled"); self.generate_button_historico.configure(state="disabled")
        self.log_textbox.configure(state="normal"); self.log_textbox.delete("1.0", "end"); self.log_textbox.configure(state="disabled")
        self.progress_bar.grid(row=2, column=0, padx=10, pady=5, sticky="ew"); self.progress_bar.set(0)
    def finalize_ui_after_generation(self):
        base_dir = self.path_entry.get()
        if base_dir: cleanup_data_folders(base_dir, self.log)
        self.progress_bar.grid_forget()
        self.generate_button_recente.configure(state="normal"); self.generate_button_historico.configure(state="normal")
        
    def start_task(self, is_historical): 
        self.prepare_ui_for_generation()
        threading.Thread(target=self.run_task, args=(is_historical,), daemon=True).start()

    def run_task(self, is_historical):
        self.after(0, plt.close, 'all')
        def thread_safe_log(msg): self.after(0, self.log, msg)
        base_dir = self.path_entry.get()
        if not base_dir: thread_safe_log("⚠️ Pasta de saída vazia."); return

        produto = self.produto_menu_historico.get() if is_historical else self.produto_menu_recente.get()
        data_dirs = {'gfs_data': os.path.join(base_dir, "gfs_data"), 'metar_data': os.path.join(base_dir, "metar_data"),
                     'satellite': os.path.join(base_dir, "satellite_data"), 'cpt': os.path.join(base_dir, "cpt_palettes"),
                     'maps': os.path.join(base_dir, "maps"), 'shapefiles': os.path.join(base_dir, "shapefiles")}
        for path in data_dirs.values(): os.makedirs(path, exist_ok=True)
        
        output_filename = None
        try:
            zoom_key = self.current_zoom
            if "Satélite" in produto:
                download_cpt_files(data_dirs['cpt'], thread_safe_log)
                download_brazil_shapefiles(data_dirs['shapefiles'], thread_safe_log)
                if is_historical:
                    date_str, time_str = self.date_menu.get(), self.time_menu.get()
                    target_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                    output_filename = plot_satellite_map(data_dirs['maps'], data_dirs, zoom_key, thread_safe_log, target_dt, 
                                                         lambda v: self.after(0, self.progress_bar.set(v)), produto, True)
                else:
                    if self.gif_checkbox_recente.get():
                        output_filename = create_satellite_gif(data_dirs['maps'], data_dirs, zoom_key, thread_safe_log, produto, lambda v: self.after(0, self.progress_bar.set(v)))
                    else:
                        output_filename = plot_satellite_map(data_dirs['maps'], data_dirs, zoom_key, thread_safe_log, produto_adicional=produto)
            else: # Lógica GFS
                region = self.region_menu_historico.get() if is_historical else self.region_menu_recente.get()
                download_brazil_shapefiles(data_dirs['shapefiles'], thread_safe_log)
                gfs_file_path, target_dt = None, None
                if is_historical:
                    date_str, time_str = self.date_menu.get(), self.time_menu.get().split(':')[0]
                    gfs_date = date_str.replace('-', '')
                    target_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
                    gfs_file_path = download_gfs_data(gfs_date, time_str, REGIONS[region], data_dirs['gfs_data'], thread_safe_log)
                else:
                    gfs_date, gfs_run = find_latest_gfs_run(thread_safe_log)
                    if gfs_date and gfs_run: gfs_file_path = download_gfs_data(gfs_date, gfs_run, REGIONS[region], data_dirs['gfs_data'], thread_safe_log)
                
                if gfs_file_path:
                    output_filename = plot_gfs_map(region, data_dirs, zoom_key, gfs_file_path, thread_safe_log, target_dt, 
                                                   lambda v: self.after(0, self.progress_bar.set(v)), produto)

            if output_filename and os.path.exists(output_filename): self.after(0, open_file, output_filename)
            thread_safe_log("\nProcesso concluído!")
        except Exception as e:
            thread_safe_log(f"\n--- ERRO INESPERADO ---\n{e}\n{traceback.format_exc()}")
        finally:
            self.after(0, self.finalize_ui_after_generation)

if __name__ == "__main__":
    if not all([StationPlot, metar, units]):
        print("AVISO: 'metpy' não encontrada. Plotagem METAR desativada.")
    warnings.filterwarnings("ignore")
    app = App()
    app.mainloop()
