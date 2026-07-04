"""
🗺️ SANKEY - Análise de LULC (Uso e Cobertura do Solo)
=======================================================
App Streamlit para visualizar transições de uso do solo (MapBiomas e outros)
com diagramas Sankey.

Análise textual AUTOMÁTICA e LOCAL (sem necessidade de API key) — funciona para
todos os usuários. Opcionalmente, se uma chave Google Gemini estiver disponível
nos *Secrets* do Streamlit (ou for informada na barra lateral), uma análise por
IA é gerada como bônus.

Autor: adaptado por Cowork/Claude
"""

import io
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
from rasterio.enums import Resampling
import streamlit as st

# Limite de pixels lidos por raster (protege o teto de ~1 GB de RAM).
# Acima disso, o raster é lido decimado (amostragem do vizinho mais próximo).
MAX_PIXELS = 20_000_000

# ----------------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SANKEY LULC", page_icon="🗺️", layout="wide")

DEFAULT_PIXEL_SIZE_M = 30.0  # fallback (MapBiomas) quando não há resolução métrica

# ----------------------------------------------------------------------------
# Utilidades de IA opcional (Gemini) — nunca obrigatória
# ----------------------------------------------------------------------------
def _get_gemini_key():
    """Procura a chave em st.secrets, variável de ambiente ou input do usuário."""
    # 1) Secrets do Streamlit (recomendado para deploy compartilhado)
    for k in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            if k in st.secrets and st.secrets[k]:
                return st.secrets[k]
        except Exception:
            pass
    # 2) Variável de ambiente
    for k in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.getenv(k):
            return os.getenv(k)
    return None


def gerar_analise_gemini(prompt, api_key):
    """Tenta gerar análise via Gemini. Retorna None em caso de qualquer falha."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        # tenta modelos em ordem de preferência
        for nome in ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"):
            try:
                model = genai.GenerativeModel(nome)
                resp = model.generate_content(prompt)
                if resp and getattr(resp, "text", None):
                    return resp.text.strip()
            except Exception:
                continue
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------------
# Leitura de QML (estilos QGIS) — suporta paletteEntry e categorized renderer
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def ler_qml(qml_bytes):
    """Extrai {valor:int -> (label, cor_hex)} de um arquivo QML."""
    classes = {}
    try:
        root = ET.fromstring(qml_bytes)
    except Exception:
        return classes

    def _norm_cor(c):
        if not c:
            return "#cccccc"
        c = c.strip()
        if c.startswith("#"):
            return c[:7]
        if "," in c:  # "r,g,b,a"
            partes = c.split(",")
            try:
                r, g, b = (int(float(x)) for x in partes[:3])
                return f"#{r:02x}{g:02x}{b:02x}"
            except Exception:
                return "#cccccc"
        return c

    def _to_int(v):
        try:
            return int(round(float(v)))
        except Exception:
            return None

    # 1) Renderer paletted / pseudocolor
    for entry in root.findall(".//paletteEntry"):
        val = _to_int(entry.get("value"))
        if val is None:
            continue
        classes[val] = (entry.get("label") or str(val), _norm_cor(entry.get("color")))

    for item in root.findall(".//colorrampshader/item"):
        val = _to_int(item.get("value"))
        if val is None:
            continue
        classes[val] = (item.get("label") or str(val), _norm_cor(item.get("color")))

    # 2) Renderer categorized (vetor-like)
    for category in root.findall(".//category"):
        val = _to_int(category.get("value"))
        if val is None:
            continue
        label = category.get("label") or str(val)
        cor = "#cccccc"
        # cor via symbol referenciado é complexo; tenta prop direto se houver
        classes.setdefault(val, (label, cor))

    return classes


# ----------------------------------------------------------------------------
# Leitura de raster TIF (detecta resolução e nodata automaticamente)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def processar_tif(tif_bytes):
    """Lê um GeoTIFF e retorna (array_1d, shape, pixel_area_ha, info)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(tif_bytes)
        path = tmp.name
    try:
        with rasterio.open(path) as src:
            h, w = src.height, src.width
            total = h * w
            nodata = src.nodata
            res_x, res_y = abs(src.res[0]), abs(src.res[1])
            crs = src.crs
            unidade_metrica = bool(crs and crs.is_projected)

            # --- Proteção de memória: decima rasters muito grandes na leitura ---
            # Cada pixel amostrado passa a representar (fator^2) pixels originais,
            # então a área por pixel é multiplicada por fator^2 (área total mantida).
            fator = 1
            if total > MAX_PIXELS:
                fator = int(math.ceil(math.sqrt(total / MAX_PIXELS)))
            if fator > 1:
                out_h, out_w = max(1, h // fator), max(1, w // fator)
                arr = src.read(1, out_shape=(out_h, out_w),
                               resampling=Resampling.nearest)  # categórico: NN
            else:
                arr = src.read(1)
            shape = (h, w)

            # --- Baixo uso de memória: manter INTEIRO, sem float64/NaN ---
            # Códigos de classe (MapBiomas etc.) são inteiros pequenos.
            # Usamos 0 como marcador de "sem dado"/fora da área.
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.nan_to_num(arr, nan=0).astype(np.int32)
            # nodata (se != 0) vira 0
            if nodata is not None and int(nodata) != 0:
                try:
                    arr[arr == int(nodata)] = 0
                except (ValueError, OverflowError):
                    pass
            # downcast para o menor inteiro sem sinal que couber (economia de RAM)
            vmax = int(arr.max()) if arr.size else 0
            if vmax <= 255:
                arr = arr.astype(np.uint8, copy=False)
            elif vmax <= 65535:
                arr = arr.astype(np.uint16, copy=False)
            else:
                arr = arr.astype(np.int32, copy=False)

            # área do pixel (ajustada pela decimação)
            if unidade_metrica and res_x > 0 and res_y > 0:
                pixel_area_m2 = res_x * res_y
                origem_res = f"{res_x:.1f}m × {res_y:.1f}m (do raster)"
            else:
                pixel_area_m2 = DEFAULT_PIXEL_SIZE_M ** 2
                origem_res = f"{DEFAULT_PIXEL_SIZE_M:.0f}m (fallback)"
            pixel_area_m2 *= fator * fator  # cada pixel lido = fator^2 originais

            pixel_area_ha = pixel_area_m2 / 10_000.0
            info = {
                "shape": shape,
                "lido": f"{arr.shape[1]}×{arr.shape[0]} px" + (
                    f" (decimado {fator}×)" if fator > 1 else ""),
                "crs": str(crs) if crs else "desconhecido",
                "res": origem_res,
                "pixel_area_ha": pixel_area_ha,
            }
            return arr, shape, pixel_area_ha, info
    finally:
        os.remove(path)


# ----------------------------------------------------------------------------
# Mapa colorido (RGB) a partir da grade 2D e das cores do QML
# ----------------------------------------------------------------------------
def mapa_rgb(grid, classes_dict, max_dim=700):
    """Converte a grade 2D de classes em imagem RGB (fundo branco onde = 0)."""
    h, w = grid.shape
    step = max(1, int(math.ceil(max(h, w) / max_dim)))
    g = grid[::step, ::step]
    rgb = np.full((g.shape[0], g.shape[1], 3), 255, dtype=np.uint8)  # fundo branco
    for cls in np.unique(g):
        if int(cls) == 0:
            continue
        _, cor = classes_dict.get(int(cls), (str(cls), "#cccccc"))
        cor = str(cor).lstrip("#")
        try:
            r, gg, b = (int(cor[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            r, gg, b = 200, 200, 200
        rgb[g == cls] = (r, gg, b)
    return rgb


def legenda_html(classes_presentes, classes_dict):
    """Monta uma legenda HTML com quadradinhos de cor para as classes presentes."""
    itens = []
    for cls in sorted(classes_presentes):
        if int(cls) == 0:
            continue
        label, cor = classes_dict.get(int(cls), (f"Classe {int(cls)}", "#cccccc"))
        itens.append(
            f'<span style="display:inline-flex;align-items:center;margin:2px 10px 2px 0">'
            f'<span style="width:14px;height:14px;background:{cor};border:1px solid #888;'
            f'display:inline-block;margin-right:5px;border-radius:2px"></span>{label}</span>'
        )
    return '<div style="display:flex;flex-wrap:wrap">' + "".join(itens) + "</div>"


# ----------------------------------------------------------------------------
# Sankey
# ----------------------------------------------------------------------------
def gerar_sankey(anos, classes_dict, transitions, pixel_area_ha, min_frac=0.0):
    """Cria a figura Sankey. min_frac filtra fluxos menores que essa fração do total."""
    labels, colors, label_map = [], [], {}
    idx = 0
    # nós por ano, na ordem de classes presentes
    classes_por_ano = []
    for j, ano in enumerate(anos):
        presentes = set()
        for t_idx, trans in enumerate(transitions):
            a1, a2 = anos[t_idx], anos[t_idx + 1]
            if a1 == ano:
                presentes.update(trans[a1].astype(int).tolist())
            if a2 == ano:
                presentes.update(trans[a2].astype(int).tolist())
        classes_por_ano.append(sorted(presentes))

    for ano, classes_ano in zip(anos, classes_por_ano):
        for cls in classes_ano:
            label, cor = classes_dict.get(int(cls), (f"Classe {cls}", "#cccccc"))
            labels.append(f"{label} ({ano})")
            colors.append(cor)
            label_map[(ano, int(cls))] = idx
            idx += 1

    total_pixels = sum(trans["count"].sum() for trans in transitions) or 1
    limite = min_frac * total_pixels

    source, target, value, link_colors, customdata = [], [], [], [], []
    for t_idx, trans in enumerate(transitions):
        a1, a2 = anos[t_idx], anos[t_idx + 1]
        # total de cada classe de origem neste período (para % de conversão)
        tot_origem = trans.groupby(a1)["count"].sum().to_dict()
        for _, row in trans.iterrows():
            c1, c2 = int(row[a1]), int(row[a2])
            if row["count"] < limite:
                continue
            if (a1, c1) in label_map and (a2, c2) in label_map:
                s = label_map[(a1, c1)]
                source.append(s)
                target.append(label_map[(a2, c2)])
                value.append(int(row["count"]))
                # link herda cor da classe de origem (com transparência)
                base = colors[s].lstrip("#")
                try:
                    r, g, b = (int(base[i:i + 2], 16) for i in (0, 2, 4))
                    link_colors.append(f"rgba({r},{g},{b},0.4)")
                except Exception:
                    link_colors.append("rgba(150,150,150,0.4)")
                base_c1 = tot_origem.get(row[a1], 0) or 1
                pct = row["count"] / base_c1 * 100.0
                customdata.append([row["count"] * pixel_area_ha, pct])

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=18, thickness=22, line=dict(color="black", width=0.4),
                  label=labels, color=colors,
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=source, target=target, value=value, color=link_colors,
                  customdata=customdata,
                  hovertemplate="%{source.label} → %{target.label}<br>"
                                "%{customdata[0]:,.1f} ha<br>"
                                "%{customdata[1]:.1f}% da classe de origem"
                                "<extra></extra>"),
    ))
    fig.update_layout(title="Transições de Uso e Cobertura do Solo (LULC)",
                      font=dict(size=12), height=720,
                      margin=dict(l=40, r=40, t=80, b=40))
    return fig


# ----------------------------------------------------------------------------
# Análise LOCAL (sem IA) — narrativa automática em português
# ----------------------------------------------------------------------------
def _nome(classes_dict, cls):
    return classes_dict.get(int(cls), (f"Classe {int(cls)}", ""))[0]


def analise_local(df, transitions, anos, classes_dict, pixel_area_ha):
    """Gera texto de análise a partir das estatísticas de transição (sem IA)."""
    total_px = len(df)
    total_ha = total_px * pixel_area_ha
    linhas = []
    linhas.append(f"**Área total analisada:** {total_ha:,.1f} ha "
                  f"({total_ha/100:,.1f} km²) · {total_px:,} pixels válidos.")
    linhas.append(f"**Períodos:** {' → '.join(anos)}.")

    # Persistência (pixels que não mudaram entre primeiro e último ano)
    a0, aN = anos[0], anos[-1]
    estaveis = int((df[a0].astype(int) == df[aN].astype(int)).sum())
    frac_est = estaveis / total_px if total_px else 0
    linhas.append(f"**Estabilidade {a0}–{aN}:** {frac_est*100:.1f}% da área manteve "
                  f"a mesma classe ({estaveis*pixel_area_ha:,.1f} ha); "
                  f"{(1-frac_est)*100:.1f}% sofreu alguma mudança.")

    # Área por classe em cada ano + variação líquida primeiro→último
    def areas_ano(ano):
        vc = df[ano].astype(int).value_counts()
        return {int(k): v * pixel_area_ha for k, v in vc.items()}

    a_ini, a_fim = areas_ano(a0), areas_ano(aN)
    todas = sorted(set(a_ini) | set(a_fim))
    variacoes = []
    for c in todas:
        ini, fim = a_ini.get(c, 0.0), a_fim.get(c, 0.0)
        variacoes.append((c, ini, fim, fim - ini))
    variacoes.sort(key=lambda x: x[3])

    ganhos = [v for v in variacoes if v[3] > 0]
    perdas = [v for v in variacoes if v[3] < 0]

    linhas.append("")
    linhas.append(f"### Variação líquida por classe ({a0} → {aN})")
    if ganhos:
        top_g = sorted(ganhos, key=lambda x: -x[3])[:5]
        linhas.append("**Maiores expansões:**")
        for c, ini, fim, d in top_g:
            pct = (d / ini * 100) if ini > 0 else float("inf")
            pct_txt = f"+{pct:.0f}%" if ini > 0 else "novo"
            linhas.append(f"- {_nome(classes_dict, c)}: {ini:,.0f} → {fim:,.0f} ha "
                          f"(**+{d:,.0f} ha**, {pct_txt})")
    if perdas:
        top_p = sorted(perdas, key=lambda x: x[3])[:5]
        linhas.append("**Maiores reduções:**")
        for c, ini, fim, d in top_p:
            pct = (d / ini * 100) if ini > 0 else 0
            linhas.append(f"- {_nome(classes_dict, c)}: {ini:,.0f} → {fim:,.0f} ha "
                          f"(**{d:,.0f} ha**, {pct:.0f}%)")

    # Principais transições (excluindo persistência) em todo o período
    fluxo = defaultdict(float)
    for t_idx, trans in enumerate(transitions):
        a1, a2 = anos[t_idx], anos[t_idx + 1]
        for _, row in trans.iterrows():
            c1, c2 = int(row[a1]), int(row[a2])
            if c1 != c2:
                fluxo[(c1, c2)] += row["count"] * pixel_area_ha
    top_trans = sorted(fluxo.items(), key=lambda x: -x[1])[:8]
    if top_trans:
        linhas.append("")
        linhas.append("### Principais transições (todas as etapas)")
        for (c1, c2), ha in top_trans:
            linhas.append(f"- {_nome(classes_dict, c1)} → {_nome(classes_dict, c2)}: "
                          f"**{ha:,.0f} ha**")

    return "\n".join(linhas), variacoes, top_trans


def montar_prompt_ia(texto_local, anos):
    return (
        "Você é um analista ambiental. Com base nas estatísticas de mudança de uso "
        "e cobertura do solo (LULC) abaixo, escreva uma análise interpretativa BREVE "
        "(3 a 5 parágrafos) em PORTUGUÊS, destacando tendências, possíveis vetores de "
        "mudança (ex.: expansão agropecuária, desmatamento, regeneração) e implicações "
        f"ambientais. Períodos: {' → '.join(anos)}.\n\n"
        "=== DADOS ===\n" + texto_local
    )


# ----------------------------------------------------------------------------
# Exportação HTML
# ----------------------------------------------------------------------------
def salvar_html(fig, analise_md, total_ha, anos):
    analise_html = analise_md.replace("\n", "<br>")
    return f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>SANKEY LULC</title>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:0;padding:24px;background:#f4f6fb}}
.container{{max-width:1400px;margin:0 auto;background:#fff;border-radius:12px;
box-shadow:0 8px 30px rgba(0,0,0,.12);padding:32px}}
h1{{color:#2c3e50}} .info{{background:#eef2ff;padding:14px;border-radius:8px;
border-left:4px solid #667eea;margin:16px 0}}
.analysis{{background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;
padding:20px;margin-top:24px;line-height:1.6}}
</style></head><body><div class="container">
<h1>🗺️ SANKEY — Análise de LULC</h1>
<div class="info"><b>Área:</b> {total_ha:,.1f} ha · <b>Períodos:</b> {' → '.join(anos)}</div>
{fig.to_html(include_plotlyjs='cdn', full_html=False)}
<div class="analysis"><h2>Análise</h2>{analise_html}</div>
</div></body></html>"""


# ============================================================================
# INTERFACE
# ============================================================================
st.title("🗺️ SANKEY — Análise de LULC")
st.caption("Transições de uso e cobertura do solo (MapBiomas e outros) com Sankey. "
           "Análise automática local — **sem necessidade de API key**.")

with st.sidebar:
    st.header("⚙️ Configurações")
    usar_ia = st.checkbox("Usar IA (Gemini) se disponível", value=True,
                          help="Se houver chave nos Secrets, gera análise por IA além da local.")
    key_manual = st.text_input("Chave Gemini (opcional)", type="password",
                               help="Deixe em branco para usar apenas a análise local.")
    min_frac_pct = st.slider("Filtrar fluxos menores que (% da área)", 0.0, 5.0, 0.0, 0.1,
                             help="Oculta transições muito pequenas para limpar o Sankey.")
    st.markdown("---")
    st.caption("💡 Para IA compartilhada no deploy, defina `GOOGLE_GEMINI_API_KEY` "
               "em *Settings → Secrets* do Streamlit Cloud.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Estilo QML (opcional)")
    qml_file = st.file_uploader("Arquivo QML do QGIS (cores e rótulos)", type=["qml"])
with col2:
    st.subheader("📊 Rasters TIF")
    tif_files = st.file_uploader("2 a 4 arquivos TIF (ordem cronológica)",
                                 type=["tif", "tiff"], accept_multiple_files=True)

anos = []
if tif_files:
    st.subheader("📅 Ano de cada arquivo")
    cols = st.columns(len(tif_files))
    for i, c in enumerate(cols):
        with c:
            st.caption(tif_files[i].name)
            ano = st.number_input(f"Ano {i+1}", min_value=1900, max_value=2100,
                                  value=1985 + i * 10, key=f"ano_{i}")
            anos.append(str(int(ano)))

pronto = tif_files and len(tif_files) >= 2

if pronto and st.button("🚀 Gerar análise", type="primary"):
    if len(set(anos)) != len(anos):
        st.error("⚠️ Os anos devem ser distintos.")
        st.stop()

    with st.spinner("Processando rasters..."):
        classes = ler_qml(qml_file.read()) if qml_file else {}
        if classes:
            st.success(f"✅ {len(classes)} classes carregadas do QML")
        else:
            st.info("ℹ️ Sem QML: usando IDs de classe como rótulos e cores padrão.")

        grids, shapes, infos, pixel_areas = [], [], [], []
        for i, tif in enumerate(tif_files):
            arr, shape, pa_ha, info = processar_tif(tif.read())  # arr = grade 2D
            grids.append(arr); shapes.append(shape)
            infos.append(info); pixel_areas.append(pa_ha)

    # valida alinhamento (pelas grades lidas, que devem ter o mesmo tamanho)
    if len({g.shape for g in grids}) != 1:
        st.error(f"⚠️ Os rasters têm dimensões diferentes: {[g.shape for g in grids]}. "
                 "Reamostre-os para a mesma grade (mesmo recorte/resolução) antes de usar.")
        st.stop()

    pixel_area_ha = pixel_areas[0]
    with st.expander("🔍 Metadados dos rasters"):
        st.dataframe(pd.DataFrame([{
            "arquivo": tif_files[i].name, "ano": anos[i], **infos[i]
        } for i in range(len(tif_files))]), width="stretch", hide_index=True)

    # máscara comum (pixels válidos = diferentes de 0 em todos os anos)
    flats = [g.ravel() for g in grids]
    mask = flats[0] != 0
    for f in flats[1:]:
        mask &= (f != 0)
    arrays = [f[mask] for f in flats]

    if len(arrays[0]) == 0:
        st.error("⚠️ Nenhum pixel válido em comum entre os rasters.")
        st.stop()

    df = pd.DataFrame({anos[i]: arrays[i] for i in range(len(arrays))})
    total_px = len(df)
    total_ha = total_px * pixel_area_ha

    c1, c2, c3 = st.columns(3)
    c1.metric("Pixels válidos", f"{total_px:,}")
    c2.metric("Hectares", f"{total_ha:,.0f}")
    c3.metric("km²", f"{total_ha/100:,.1f}")

    # transições consecutivas
    transitions = []
    for i in range(len(anos) - 1):
        t = df.groupby([anos[i], anos[i + 1]]).size().reset_index(name="count")
        transitions.append(t)

    # Sankey
    st.subheader("📊 Diagrama Sankey")
    st.caption("Passe o mouse sobre os fluxos para ver a área (ha) e a "
               "**% de conversão** em relação à classe de origem.")
    fig = gerar_sankey(anos, classes, transitions, pixel_area_ha,
                       min_frac=min_frac_pct / 100.0)
    st.plotly_chart(fig, width="stretch")

    # Mapas por ano
    st.subheader("🗺️ Mapas por ano")
    presentes = set()
    for g in grids:
        presentes.update(int(v) for v in np.unique(g))
    st.markdown(legenda_html(presentes, classes), unsafe_allow_html=True)
    mcols = st.columns(len(grids))
    for i, col in enumerate(mcols):
        with col:
            st.image(mapa_rgb(grids[i], classes), caption=anos[i], width="stretch")

    # Análise local
    st.subheader("📝 Análise automática (local)")
    texto_local, variacoes, top_trans = analise_local(
        df, transitions, anos, classes, pixel_area_ha)
    st.markdown(texto_local)

    analise_final = texto_local

    # Análise IA opcional
    if usar_ia:
        api_key = key_manual.strip() or _get_gemini_key()
        if api_key:
            with st.spinner("Gerando análise com IA (Gemini)..."):
                ia = gerar_analise_gemini(montar_prompt_ia(texto_local, anos), api_key)
            if ia:
                st.subheader("🤖 Análise interpretativa (IA)")
                st.markdown(ia)
                analise_final = texto_local + "\n\n---\n\n### Análise por IA\n\n" + ia
            else:
                st.caption("IA indisponível no momento — exibindo apenas a análise local.")
        else:
            st.caption("Nenhuma chave Gemini configurada — usando apenas a análise local.")

    # Tabela de variação por classe
    st.subheader("📈 Variação por classe")
    df_var = pd.DataFrame(
        [(_nome(classes, c), ini, fim, d) for c, ini, fim, d in variacoes],
        columns=["Classe", f"{anos[0]} (ha)", f"{anos[-1]} (ha)", "Δ (ha)"]
    ).sort_values("Δ (ha)")
    st.dataframe(df_var, width="stretch", hide_index=True)

    # Tabelas de transição detalhadas
    st.subheader("📋 Tabelas de transição")
    export_frames = []
    for i, trans in enumerate(transitions):
        disp = trans.copy()
        # % de conversão em relação ao total da classe de origem no período
        tot_origem = disp.groupby(anos[i])["count"].transform("sum")
        disp["% origem"] = (disp["count"] / tot_origem * 100).round(1)
        disp["De"] = disp[anos[i]].apply(lambda x: _nome(classes, x))
        disp["Para"] = disp[anos[i + 1]].apply(lambda x: _nome(classes, x))
        disp["Hectares"] = disp["count"] * pixel_area_ha
        disp["km²"] = disp["Hectares"] / 100
        disp = disp[["De", "Para", "count", "% origem", "Hectares", "km²"]].rename(
            columns={"count": "Pixels"}).sort_values("Hectares", ascending=False)
        disp.insert(0, "Período", f"{anos[i]}→{anos[i+1]}")
        export_frames.append(disp)
        with st.expander(f"{anos[i]} → {anos[i+1]}"):
            st.dataframe(disp.drop(columns=["Período"]),
                         width="stretch", hide_index=True)

    # Downloads
    st.subheader("📥 Downloads")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("📄 HTML completo",
                           salvar_html(fig, analise_final, total_ha, anos),
                           file_name="sankey_lulc.html", mime="text/html")
    with d2:
        csv = pd.concat(export_frames, ignore_index=True).to_csv(index=False).encode("utf-8")
        st.download_button("📊 Transições (CSV)", csv,
                           file_name="transicoes_lulc.csv", mime="text/csv")
    with d3:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df_var.to_excel(xw, sheet_name="Variacao_classe", index=False)
            pd.concat(export_frames, ignore_index=True).to_excel(
                xw, sheet_name="Transicoes", index=False)
        st.download_button("📗 Excel (XLSX)", buf.getvalue(),
                           file_name="analise_lulc.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif not pronto:
    st.info("👈 Envie pelo menos **2 arquivos TIF** para começar. O QML é opcional "
            "(melhora cores e rótulos).")
