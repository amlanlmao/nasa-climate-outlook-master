NASA Climate Outlook — theme.py update

Replace your existing project file:
    core/theme.py

with this theme.py.

The update preserves the existing public function names:
- inject_theme()
- header(title, subtitle)
- warming_stripes(series)
- style_fig(fig, height=None)
- plot(fig, height=None, **kwargs)
- line_fig(data, ylabel)
- importance_fig(values, xlabel)

No app.py changes are required if your app already calls these functions.
Restart Streamlit after replacing the file:
    python -m streamlit run app.py

Google Fonts are loaded from Google Fonts when internet access is available;
system fallback fonts are included.
