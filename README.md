# 🗺️ SANKEY — Análise de LULC (Uso e Cobertura do Solo)

Visualize e analise mudanças de uso e cobertura do solo (LULC) a partir de rasters
(MapBiomas e outros) com diagramas **Sankey**. A análise textual é gerada
**automaticamente e localmente — sem necessidade de API key** — para que qualquer
pessoa possa usar o app no Streamlit. A IA (Google Gemini) é **opcional** e entra
apenas como bônus quando uma chave está disponível.

## ✨ Novidades desta versão

- **Sem API key obrigatória.** Análise estatística/textual em português gerada por
  regras (estabilidade, variação líquida por classe, principais transições).
- **IA opcional (Gemini).** Se houver chave nos *Secrets* do Streamlit, o app gera
  também uma análise interpretativa. Nenhum usuário precisa digitar chave.
- **Resolução automática.** A área do pixel é lida do metadata de cada GeoTIFF
  (fallback de 30 m quando o CRS não é métrico), em vez de assumir 30 m fixo.
- **QML mais robusto** (paletteEntry, colorrampshader e categorized).
- **Validação de alinhamento** dos rasters e tratamento de nodata.
- **Cache** de leitura para respostas rápidas.
- **Exportações**: HTML completo, CSV de transições e Excel (XLSX).
- Filtro opcional para ocultar fluxos muito pequenos e limpar o Sankey.

## 🚀 Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre em `http://localhost:8501`. O app funciona sem nenhuma configuração extra.

## ☁️ Deploy no Streamlit Community Cloud (todos usam sem chave)

1. Suba estes arquivos para um repositório GitHub (`app.py`, `requirements.txt`,
   `packages.txt`, `README.md`, `.gitignore`).
2. Em [share.streamlit.io](https://share.streamlit.io), aponte para `app.py`.
3. **(Opcional)** Para habilitar a IA compartilhada: App → **Settings → Secrets** e
   cole o conteúdo de `secrets.toml.example` com sua chave Gemini. Assim todos os
   visitantes recebem a análise por IA sem digitar nada. Sua cota gratuita do Gemini
   é usada. Sem isso, todos ainda têm a análise local completa.

`packages.txt` instala o GDAL do sistema, necessário para o `rasterio` no Cloud.

## 🔑 Chave Gemini (opcional)

Grátis em <https://aistudio.google.com/apikey>. Ordem de busca da chave:
`st.secrets` → variável de ambiente (`GOOGLE_GEMINI_API_KEY`) → campo na barra lateral.

> ⚠️ **Segurança:** nunca faça commit da chave. Use `Secrets`/`.env`. Se uma chave
> foi exposta em arquivo, revogue-a e gere outra.

## 📥 Como usar

1. **(Opcional)** Envie um `.qml` do QGIS para cores e rótulos das classes.
2. Envie **2 a 4 GeoTIFFs** classificados (ordem cronológica).
3. Informe o **ano** de cada arquivo.
4. Clique em **Gerar análise**: Sankey, análise automática, tabelas e downloads.

## 🧱 Estrutura

```
├── app.py                 # App Streamlit principal
├── requirements.txt       # Dependências Python
├── packages.txt           # Dependências de sistema (GDAL) p/ Streamlit Cloud
├── secrets.toml.example   # Modelo de Secrets (IA opcional)
├── .gitignore
└── README.md
```

Disponivel em:
https://webbaseflowseparator.streamlit.app/

## 📝 Notas técnicas

- Os rasters precisam estar na **mesma grade** (mesmo recorte e resolução). O app
  avisa se as dimensões diferirem — reamostre no QGIS/GDAL antes.
- Pixels com valor `0` ou `nodata` são tratados como fora da área.
- A área é calculada com a máscara comum de pixels válidos em todos os anos.
