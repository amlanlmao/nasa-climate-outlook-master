"""Apply the new look to your Streamlit dashboard.

Usage:  python apply_theme.py path/to/app.py
Keep theme.py next to this script. It is copied to <app folder>/core/theme.py,
your app is backed up to app.py.bak, and every edit is reported.
"""
import ast
import re
import shutil
import sys
from pathlib import Path

app = Path(sys.argv[1] if len(sys.argv) > 1 else "app.py").resolve()
root = app.parent
src = app.read_text(encoding="utf-8")
shutil.copy(app, app.with_name(app.name + ".bak"))
(root / "core").mkdir(exist_ok=True)
shutil.copy(Path(__file__).with_name("theme.py"), root / "core" / "theme.py")

cfg = root / ".streamlit" / "config.toml"
if not cfg.exists():
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text('[theme]\nprimaryColor = "#1f5fa6"\nfont = "sans serif"\n')


def swap(label, old, new):
    global src
    if old in src:
        src = src.replace(old, new)
        print(f"  applied  {label}")
    else:
        print(f"  SKIPPED  {label} (text not found)")


swap("imports",
     "from core.explainability import explain_random_forest\n",
     "from core.explainability import explain_random_forest\n"
     "from core.theme import (\n    BLUE, GREY, RED,\n"
     "    header, importance_fig, inject_theme, line_fig, plot, warming_stripes,\n)\n")

start = src.find('st.markdown(\n    """\n    <style>')
end = src.find("# ---------------------------------------------------------------------\n# Sidebar controls")
if 0 <= start < end:
    src = (src[:start] + "inject_theme()\nheader(\n    \"NASA Climate Outlook\",\n"
           "    \"Global temperature monitoring, historical exploration, experimental \"\n"
           "    \"forecasting and regional climate analysis.\",\n)\n\n\n" + src[end:])
    print("  applied  header and CSS")
else:
    print("  SKIPPED  header and CSS (text not found)")

swap("overview label",
     "    st.markdown(\n        '<div class=\"section-label\">Current conditions and outlook</div>',\n"
     "        unsafe_allow_html=True,\n    )\n\n", "")
swap("direction arrows",
     '    metric_columns[4].metric("Direction", direction)',
     '    metric_columns[4].metric(\n        "Direction",\n'
     '        {"Rising": "▲ ", "Falling": "▼ ", "Stable": "● "}[direction] + direction,\n    )')
swap("baseline bar chart",
     '    st.bar_chart(\n        baseline_comparison.set_index("Model")["MAE (°C)"],\n        color="#2563eb",\n    )',
     '    mae_fig = px.bar(\n        baseline_comparison, x="MAE (°C)", y="Model", orientation="h",\n'
     '        text_auto=".3f", color="Model", color_discrete_sequence=[BLUE, GREY],\n    )\n'
     '    mae_fig.update_layout(showlegend=False, yaxis_title=None)\n    plot(mae_fig, height=220)')
swap("regional chart",
     '        st.line_chart(regional, y_label="Anomaly (°C)")',
     '        plot(line_fig(regional, "Anomaly (°C)"), height=380)')
swap("SHAP chart",
     '        st.bar_chart(\n            shap_values,\n            y_label="Mean absolute SHAP value",\n        )',
     '        plot(importance_fig(shap_values, "Mean absolute SHAP value"))')
swap("warming stripes",
     "plot_observed = observed.iloc[-history_months:]\n",
     "plot_observed = observed.iloc[-history_months:]\n\nwarming_stripes(series)\n")

src, n = re.subn(r"st\.plotly_chart\(", "plot(", src)
print(f"  applied  styled {n} Plotly charts")
for old, new in [('"#2563eb"', "BLUE"), ('"#dc2626"', "RED"),
                 ("rgba(220,38,38,", "rgba(214,64,42,"), ("rgba(37,99,235,", "rgba(31,95,166,")]:
    src = src.replace(old, new)

ast.parse(src)
app.write_text(src, encoding="utf-8")
print(f"\nDone. Backup saved as {app.name}.bak")
