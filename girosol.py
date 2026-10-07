"""
GiroSol — Simulação comparativa: painel fotovoltaico fixo × rastreador solar de 1 eixo.

Modelo
------
- Local: Pelotas/RS.
- Painel convencional: fixo, inclinação ≈ latitude local, voltado ao Norte.
- GiroSol: rastreador de 1 eixo (eixo Norte-Sul, giro Leste-Oeste), com backtracking.
- Irradiância: céu claro (pvlib / Ineichen). NÃO é medição meteorológica real.
- Os dois painéis recebem exatamente o mesmo recurso solar.

Instalação
----------
    pip install pvlib numpy pandas matplotlib
    (no Linux, o tkinter vem de: sudo apt install python3-tk)

Execução
--------
    python girosol.py
"""

from __future__ import annotations

import inspect
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pvlib import irradiance, tracking
from pvlib.location import Location


# ============================================================
# CONFIGURAÇÕES
# ============================================================

LATITUDE = -31.77
LONGITUDE = -52.34
ALTITUDE = 17
TIMEZONE = "America/Sao_Paulo"

STEP_MIN = 5  # intervalo entre pontos da simulação (minutos)

# Painel convencional: inclinado pela latitude, voltado ao Norte (azimute 0°).
FIXED_TILT = abs(LATITUDE)
FIXED_AZIMUTH = 0.0

# GiroSol: rastreador de 1 eixo, eixo Norte-Sul, giro Leste-Oeste.
AXIS_TILT = 0.0
AXIS_AZIMUTH = 0.0
MAX_ANGLE = 60.0
BACKTRACK = True
GCR = 0.35

# Valores iniciais dos campos
DEFAULT_AREA_M2 = 1.00
DEFAULT_EFFICIENCY_PCT = 20

# O pvlib renomeou 'apparent_azimuth' para 'solar_azimuth' (versões novas).
_AZIMUTH_KW = ("solar_azimuth"
               if "solar_azimuth" in inspect.signature(tracking.singleaxis).parameters
               else "apparent_azimuth")

LOCATION = Location(
    latitude=LATITUDE,
    longitude=LONGITUDE,
    tz=TIMEZONE,
    altitude=ALTITUDE,
    name="Pelotas - RS",
)

# Cores
NAVY = "#0B2545"
BG = "#F3F5F8"
CARD = "#FFFFFF"
TEXT = "#1E293B"
MUTED = "#64748B"
CONV_COLOR = "#3B6EA8"
GIRO_COLOR = "#E8590C"
SUN_COLOR = "#FDB813"
GREEN = "#15803D"

# Velocidades da animação: nome -> (pontos avançados por quadro, atraso em ms)
SPEEDS = {
    "Lenta": (1, 90),
    "Normal": (1, 45),
    "Rápida": (3, 40),
}


# ============================================================
# MOTOR DA SIMULAÇÃO
# ============================================================

def _clear_sky(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Céu claro (Ineichen). Se a tabela de turbidez falhar, usa um valor típico."""
    try:
        return LOCATION.get_clearsky(times, model="ineichen")
    except Exception:
        return LOCATION.get_clearsky(times, model="ineichen", linke_turbidity=3.0)


def _local_timestamp(day, hour: int) -> pd.Timestamp:
    ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    return ts.tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous=True)


def _cumulative_wh(power_w: np.ndarray, step_h: float) -> np.ndarray:
    """Energia acumulada (Wh) por regra do trapézio."""
    energy = np.zeros_like(power_w, dtype=float)
    if len(power_w) > 1:
        energy[1:] = np.cumsum((power_w[1:] + power_w[:-1]) / 2.0 * step_h)
    return energy


def simulate(day, start_hour: int, end_hour: int,
             area_m2: float = DEFAULT_AREA_M2,
             efficiency: float = DEFAULT_EFFICIENCY_PCT / 100) -> pd.DataFrame:
    """
    Simula o dia e devolve um DataFrame com uma linha por instante.

    Potência de cada painel:  P = POA × A × η
    onde POA é a irradiância que de fato incide no plano do painel
    (já inclui o efeito do ângulo de incidência θ e a parcela difusa).
    """
    times = pd.date_range(
        _local_timestamp(day, start_hour),
        _local_timestamp(day, end_hour),
        freq=f"{STEP_MIN}min",
    )

    sun = LOCATION.get_solarposition(times)
    clear = _clear_sky(times)
    zenith = sun["apparent_zenith"]
    azimuth = sun["azimuth"]
    sun_is_up = (sun["apparent_elevation"] > 0).to_numpy()

    # --- Painel convencional (ângulo fixo) ---
    fixed_tilt = pd.Series(FIXED_TILT, index=times)
    fixed_az = pd.Series(FIXED_AZIMUTH, index=times)

    # --- GiroSol (rastreador de 1 eixo) ---
    track = tracking.singleaxis(
        apparent_zenith=zenith,
        axis_tilt=AXIS_TILT,
        axis_azimuth=AXIS_AZIMUTH,
        max_angle=MAX_ANGLE,
        backtrack=BACKTRACK,
        gcr=GCR,
        **{_AZIMUTH_KW: azimuth},
    )
    # À noite o rastreador devolve NaN: deixamos o painel na horizontal.
    giro_tilt = track["surface_tilt"].fillna(0.0)
    giro_az = track["surface_azimuth"].fillna(90.0)

    def poa(tilt: pd.Series, az: pd.Series) -> np.ndarray:
        total = irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=az,
            solar_zenith=zenith,
            solar_azimuth=azimuth,
            dni=clear["dni"],
            ghi=clear["ghi"],
            dhi=clear["dhi"],
        )
        return total["poa_global"].fillna(0.0).clip(lower=0.0).to_numpy() * sun_is_up

    def incidence(tilt: pd.Series, az: pd.Series) -> np.ndarray:
        return irradiance.aoi(tilt, az, zenith, azimuth).fillna(90.0).to_numpy()

    fixed_poa = poa(fixed_tilt, fixed_az)
    giro_poa = poa(giro_tilt, giro_az)

    fixed_power = fixed_poa * area_m2 * efficiency
    giro_power = giro_poa * area_m2 * efficiency
    step_h = STEP_MIN / 60.0

    return pd.DataFrame({
        "time": times,
        "hour": times.hour + times.minute / 60.0,
        "sun_elevation": sun["apparent_elevation"].to_numpy(),
        "sun_azimuth": azimuth.to_numpy(),
        "ghi": clear["ghi"].to_numpy(),
        "dni": clear["dni"].to_numpy(),
        "dhi": clear["dhi"].to_numpy(),
        "fixed_tilt": fixed_tilt.to_numpy(),
        "fixed_azimuth": fixed_az.to_numpy(),
        "giro_tilt": giro_tilt.to_numpy(),
        "giro_azimuth": giro_az.to_numpy(),
        "fixed_aoi": incidence(fixed_tilt, fixed_az),
        "giro_aoi": incidence(giro_tilt, giro_az),
        "fixed_poa": fixed_poa,
        "giro_poa": giro_poa,
        "fixed_power": fixed_power,
        "giro_power": giro_power,
        "fixed_energy": _cumulative_wh(fixed_power, step_h),
        "giro_energy": _cumulative_wh(giro_power, step_h),
    })


# ============================================================
# UTILIDADES DE FORMATAÇÃO
# ============================================================

def br(value: float, decimals: int = 1) -> str:
    """Formata número no padrão brasileiro (1.234,5)."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def hour_label(hour: float, _pos=None) -> str:
    minutes = int(round(hour * 60))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ============================================================
# CENA VISUAL (painel + Sol em pseudo-3D)
# ============================================================
# Coordenadas 3D: E = Leste, N = Norte, U = Cima.
# Visão: de frente para o Sul, ou seja, Leste à esquerda e Oeste à direita
# (o Sol nasce à esquerda e se põe à direita, como no mockup).

PANEL_HEIGHT = 0.45   # altura do eixo do painel
PANEL_HL = 0.60       # meio-comprimento (ao longo do eixo de rotação)
PANEL_HW = 0.42       # meia-largura
SUN_DISTANCE = 1.25


def _project(p) -> tuple[float, float]:
    e, n, u = p
    return (-e - 0.55 * n, u - 0.30 * n)


def _panel_corners(tilt_deg: float, azimuth_deg: float, axis: str):
    """Cantos do painel em 3D. axis: 'EW' (fixo) ou 'NS' (rastreador)."""
    t, a = np.radians(tilt_deg), np.radians(azimuth_deg)
    normal = np.array([np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)])
    u = np.array([1.0, 0.0, 0.0]) if axis == "EW" else np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    c = np.array([0.0, 0.0, PANEL_HEIGHT])
    return [c - PANEL_HL * u - PANEL_HW * v,
            c + PANEL_HL * u - PANEL_HW * v,
            c + PANEL_HL * u + PANEL_HW * v,
            c - PANEL_HL * u + PANEL_HW * v]


def _sun_vector(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def _sky_color(elevation: float):
    stops = [-6, 0, 12, 35]
    colors = np.array([to_rgb(c) for c in ("#12203B", "#F2A65A", "#9ECFEA", "#BFE6FA")])
    return tuple(float(np.interp(elevation, stops, colors[:, i])) for i in range(3))


def _blend(c1: str, c2: str, k: float):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * float(np.clip(k, 0, 1)))


class PanelScene:
    """Um painel + o Sol, desenhados uma única vez e apenas atualizados a cada quadro."""

    def __init__(self, ax, title: str, color: str, axis: str):
        self.ax, self.color, self.axis = ax, color, axis

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-0.45, 1.55)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=6)

        # Chão
        ground = [_project((e, n, 0)) for e, n in
                  [(-1.5, -0.7), (1.5, -0.7), (1.5, 0.7), (-1.5, 0.7)]]
        ax.add_patch(Polygon(ground, closed=True, fc="#8FAE72", ec="#6B8A52", lw=1, zorder=1))

        # Pontos cardeais
        for label, pos in (("LESTE", (1.15, 0.5, 0)), ("OESTE", (-1.15, 0.5, 0)),
                           ("NORTE", (0.0, 0.88, 0))):
            x, y = _project(pos)
            ax.text(x, y, label, fontsize=7, color="#3F5B2E", ha="center", va="center", zorder=2)

        # Poste
        (x0, y0), (x1, y1) = _project((0, 0, 0)), _project((0, 0, PANEL_HEIGHT))
        ax.plot([x0, x1], [y0, y1], lw=4, color="#475569", solid_capstyle="round", zorder=2)

        # Painel, raio, Sol
        self.panel = Polygon(np.zeros((4, 2)), closed=True, ec=color, lw=2.2, zorder=3)
        ax.add_patch(self.panel)
        (self.ray,) = ax.plot([], [], ls="--", lw=1.2, color="#F59E0B", alpha=0.8, zorder=4)
        self.glow = Circle((0, 0), 0.25, fc=SUN_COLOR, alpha=0.30, ec="none", zorder=5)
        self.sun = Circle((0, 0), 0.14, fc=SUN_COLOR, ec="#F59E0B", lw=2, zorder=6)
        ax.add_patch(self.glow)
        ax.add_patch(self.sun)

    def update(self, sun_elevation, sun_azimuth, tilt, surface_azimuth, aoi_deg):
        self.ax.set_facecolor(_sky_color(sun_elevation))
        self.panel.set_xy([_project(c) for c in _panel_corners(tilt, surface_azimuth, self.axis)])

        sun_up = sun_elevation > 0
        lit = np.cos(np.radians(aoi_deg)) if sun_up else 0.0
        self.panel.set_facecolor(_blend("#1B2A44", "#4F9BFF", lit))

        for artist in (self.sun, self.glow, self.ray):
            artist.set_visible(bool(sun_up))
        if sun_up:
            sx, sy = _project(SUN_DISTANCE * _sun_vector(sun_elevation, sun_azimuth))
            self.sun.center = (sx, sy)
            self.glow.center = (sx, sy)
            cx, cy = _project((0, 0, PANEL_HEIGHT))
            self.ray.set_data([cx, sx], [cy, sy])


# ============================================================
# GRÁFICO DE POTÊNCIA
# ============================================================

class PowerChart:
    def __init__(self, ax):
        self.ax = ax
        self.cursor = self.dot_fixed = self.dot_giro = None

    def plot(self, df: pd.DataFrame):
        ax, x = self.ax, df["hour"].to_numpy()
        ax.clear()

        ax.fill_between(x, df["fixed_power"], color=CONV_COLOR, alpha=0.12)
        ax.fill_between(x, df["giro_power"], color=GIRO_COLOR, alpha=0.12)
        ax.plot(x, df["fixed_power"], color=CONV_COLOR, lw=2.2, label="Convencional (fixo)")
        ax.plot(x, df["giro_power"], color=GIRO_COLOR, lw=2.2, label="GiroSol (1 eixo)")

        self.cursor = ax.axvline(x[0], color="#94A3B8", lw=1.2, ls="--")
        (self.dot_fixed,) = ax.plot([x[0]], [df["fixed_power"].iloc[0]], "o",
                                    color=CONV_COLOR, ms=8, mec="white", zorder=5)
        (self.dot_giro,) = ax.plot([x[0]], [df["giro_power"].iloc[0]], "o",
                                   color=GIRO_COLOR, ms=8, mec="white", zorder=5)

        peak = max(df["fixed_power"].max(), df["giro_power"].max(), 1.0)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, peak * 1.18)
        ax.xaxis.set_major_locator(MultipleLocator(1 if x[-1] - x[0] <= 12 else 2))
        ax.xaxis.set_major_formatter(FuncFormatter(hour_label))
        ax.set_title("Potência ao longo do dia", fontsize=11, fontweight="bold",
                     color=TEXT, loc="left")
        ax.set_xlabel("Horário", color=MUTED)
        ax.set_ylabel("Potência (W)", color=MUTED)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", frameon=False)

    def move_cursor(self, hour: float, fixed_power: float, giro_power: float):
        self.cursor.set_xdata([hour, hour])
        self.dot_fixed.set_data([hour], [fixed_power])
        self.dot_giro.set_data([hour], [giro_power])



# ============================================================
# TABELAS
# ============================================================

def hourly_table(df: pd.DataFrame):
    """Energia gerada por faixa de 1 hora: (faixa, GHI média, Wh fixo, Wh GiroSol)."""
    marks = df.index[(df["hour"] % 1 == 0)].to_list()
    rows = []
    for a, b in zip(marks[:-1], marks[1:]):
        rows.append((
            f"{hour_label(df.at[a, 'hour'])} – {hour_label(df.at[b, 'hour'])}",
            df.loc[a:b, "ghi"].mean(),
            df.at[b, "fixed_energy"] - df.at[a, "fixed_energy"],
            df.at[b, "giro_energy"] - df.at[a, "giro_energy"],
        ))
    return rows


def comparison_rows(gain_pct: float | None):
    """Tabela comparativa: painel solar fixo × painel solar móvel (GiroSol)."""
    gain = (f" Na simulação: {gain_pct:+.1f}".replace(".", ",") + "% de energia."
            if gain_pct is not None else "")
    return [
        ("Mobilidade",
         "Não se move: fica em um único ângulo (aqui, ≈ a latitude, voltado ao Norte).",
         "Gira em 1 eixo (Leste–Oeste) por motor, acompanhando o Sol ao longo do dia."),
        ("Aproveitamento da luz solar",
         "Melhor por volta do meio-dia; de manhã e à tarde os raios chegam inclinados e a captação cai.",
         "Fica mais voltado ao Sol, com produção alta e estável por mais horas." + gain),
        ("Possibilidade de ajuste",
         "Só manual e ocasional (ex.: mudar a inclinação por estação do ano).",
         "Ajuste automático e contínuo, por sensores ou cálculo da posição do Sol."),
        ("Aplicações",
         "Telhados de casas e comércios, locais pequenos ou de difícil acesso.",
         "Usinas solares, áreas planas e abertas, projetos em que o ganho compensa o custo."),
        ("Limitações",
         "Gera menos fora do horário de pico e não se adapta ao movimento do Sol.",
         "Custo maior, manutenção de motor e peças móveis, consumo próprio, "
         "sombra entre fileiras e vento; ganho menor em dia nublado."),
        ("Impacto social",
         "Tecnologia simples e acessível, que popularizou a energia solar nas residências.",
         "Mais energia limpa por área e menor custo do kWh em grande escala; "
         "pode ser cara para famílias de baixa renda."),
        ("Potencial de melhoria",
         "Células mais eficientes, ângulo otimizado e ajuste sazonal.",
         "Sensores, previsão do tempo, 2 eixos, motores mais eficientes e soluções de baixo custo."),
    ]


# ============================================================
# INTERFACE
# ============================================================

class GiroSolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.df: pd.DataFrame | None = None
        self.frame_idx = 0
        self.playing = False
        self._after_id = None
        self._slider_lock = False

        self._setup_window()
        self._build_style()
        self._build_header()
        self._build_controls()
        # Os blocos de baixo são empacotados primeiro para nunca serem "cortados".
        self._build_summary()
        self._build_cards()
        self._build_playback()
        self._build_figure()

        self.run_simulation()

    # ---------- janela e estilo ----------

    def _setup_window(self):
        self.root.title("GiroSol — Simulação de Rastreamento Solar")
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(1280, sw - 40), min(900, sh - 80)
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 20)}")
        self.root.minsize(980, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self):
        families = set(tkfont.families())
        family = next((f for f in ("Segoe UI", "Helvetica Neue", "Ubuntu", "DejaVu Sans", "Arial")
                       if f in families), tkfont.nametofont("TkDefaultFont").actual("family"))
        self.font_family = family

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base = (family, 10)
        bold = (family, 10, "bold")

        style.configure(".", font=base, background=BG, foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("TLabelframe", background=BG, bordercolor="#CBD5E1")
        style.configure("TLabelframe.Label", background=BG, foreground=NAVY, font=bold)

        style.configure("TButton", padding=(12, 6))
        style.configure("Accent.TButton", background=GIRO_COLOR, foreground="white",
                        font=bold, padding=(16, 6), borderwidth=0)
        style.map("Accent.TButton",
                  background=[("pressed", "#9A3412"), ("active", "#C2410C")])

        style.configure("Card.TFrame", background=CARD, relief="solid",
                        borderwidth=1, bordercolor="#D5DCE5")
        style.configure("Card.TLabel", background=CARD)
        style.configure("CardSub.TLabel", background=CARD, foreground=MUTED, font=(family, 9))
        style.configure("Big.TLabel", background=CARD, font=(family, 20, "bold"))
        for name, color in (("Conv", CONV_COLOR), ("Giro", GIRO_COLOR), ("Sun", "#B45309")):
            style.configure(f"{name}.CardTitle.TLabel", background=CARD,
                            foreground=color, font=bold)

        style.configure("Horizontal.TScale", background=BG, troughcolor="#CBD5E1")
        style.configure("Result.TLabel", font=(family, 11, "bold"))
        style.configure("Gain.TLabel", font=(family, 11, "bold"), foreground=GREEN)
        style.configure("Note.TLabel", foreground=MUTED, font=(family, 9))
        style.configure("Time.TLabel", font=(family, 13, "bold"), foreground=NAVY)

    # ---------- blocos da interface ----------

    def _build_header(self):
        bar = tk.Frame(self.root, bg=NAVY)
        bar.pack(fill="x")

        left = tk.Frame(bar, bg=NAVY)
        left.pack(side="left", padx=22, pady=10)
        tk.Label(left, text="☀", bg=NAVY, fg=SUN_COLOR,
                 font=(self.font_family, 26, "bold")).pack(side="left", padx=(0, 10))
        titles = tk.Frame(left, bg=NAVY)
        titles.pack(side="left")
        tk.Label(titles, text="GIROSOL", bg=NAVY, fg="white",
                 font=(self.font_family, 20, "bold")).pack(anchor="w")
        tk.Label(titles, text="Painel convencional × rastreador solar de 1 eixo",
                 bg=NAVY, fg="#B6C4D9", font=(self.font_family, 10)).pack(anchor="w")

        right = tk.Frame(bar, bg=NAVY)
        right.pack(side="right", padx=22)
        tk.Label(right, text="PELOTAS / RS", bg=NAVY, fg="white",
                 font=(self.font_family, 12, "bold")).pack(anchor="e")
        tk.Label(right, text=f"{abs(LATITUDE):.2f}° S · {abs(LONGITUDE):.2f}° O  •  "
                             "eixo N-S  •  céu claro".replace(".", ","),
                 bg=NAVY, fg="#B6C4D9", font=(self.font_family, 9)).pack(anchor="e")

    def _build_controls(self):
        frame = ttk.LabelFrame(self.root, text="Parâmetros da simulação", padding=(12, 8))
        frame.pack(fill="x", padx=20, pady=(12, 8))

        fields = ttk.Frame(frame)
        fields.pack(side="left")

        self.date_var = tk.StringVar(value=datetime.now().strftime("%d/%m/%Y"))
        self.start_var = tk.StringVar(value="08")
        self.end_var = tk.StringVar(value="18")
        self.area_var = tk.StringVar(value=br(DEFAULT_AREA_M2, 2))
        self.eff_var = tk.StringVar(value=str(DEFAULT_EFFICIENCY_PCT))

        entries = []

        def add_field(col: int, label: str, widget: ttk.Widget):
            ttk.Label(fields, text=label).grid(row=0, column=col * 2, sticky="e",
                                               padx=(0 if col == 0 else 14, 5))
            widget.grid(row=0, column=col * 2 + 1)

        date_entry = ttk.Entry(fields, textvariable=self.date_var, width=11, justify="center")
        add_field(0, "Data", date_entry)
        add_field(1, "Início (h)", ttk.Combobox(
            fields, textvariable=self.start_var, width=4, state="readonly",
            values=[f"{h:02d}" for h in range(4, 20)], justify="center"))
        add_field(2, "Fim (h)", ttk.Combobox(
            fields, textvariable=self.end_var, width=4, state="readonly",
            values=[f"{h:02d}" for h in range(5, 22)], justify="center"))
        area_entry = ttk.Entry(fields, textvariable=self.area_var, width=7, justify="center")
        add_field(3, "Área (m²)", area_entry)
        eff_entry = ttk.Entry(fields, textvariable=self.eff_var, width=6, justify="center")
        add_field(4, "Eficiência (%)", eff_entry)

        for entry in (date_entry, area_entry, eff_entry):
            entry.bind("<Return>", lambda _e: self.run_simulation())

        buttons = ttk.Frame(frame)
        buttons.pack(side="right")
        ttk.Button(buttons, text="Tabelas", command=self.show_tables).pack(
            side="right", padx=(8, 0))
        ttk.Button(buttons, text="Exportar CSV", command=self.export_csv).pack(
            side="right", padx=(8, 0))
        ttk.Button(buttons, text="▶  Simular", style="Accent.TButton",
                   command=self.run_simulation).pack(side="right")

    def _build_summary(self):
        frame = ttk.LabelFrame(self.root, text="Resultado do período", padding=(12, 6))
        frame.pack(side="bottom", fill="x", padx=20, pady=(4, 12))

        self.fixed_result = ttk.Label(frame, text="Convencional: —", style="Result.TLabel")
        self.giro_result = ttk.Label(frame, text="GiroSol: —", style="Result.TLabel")
        self.diff_result = ttk.Label(frame, text="Ganho do GiroSol: —", style="Gain.TLabel")
        for label in (self.fixed_result, self.giro_result, self.diff_result):
            label.pack(side="left", padx=(0, 28))

        ttk.Label(frame, text="Céu claro (modelo) — não é medição real",
                  style="Note.TLabel").pack(side="right")

    def _build_cards(self):
        row = ttk.Frame(self.root)
        row.pack(side="bottom", fill="x", padx=20, pady=(4, 4))
        for col in range(3):
            row.columnconfigure(col, weight=1, uniform="cards")

        self.card_fixed = self._make_card(row, 0, "CONVENCIONAL", "Conv")
        self.card_giro = self._make_card(row, 1, "GIROSOL", "Giro")
        self.card_sun = self._make_card(row, 2, "SOL — IRRADIÂNCIA GLOBAL", "Sun")

    def _make_card(self, parent, col: int, title: str, kind: str) -> dict[str, tk.StringVar]:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 8))
        card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))

        vars_ = {k: tk.StringVar(value="—") for k in ("big", "l1", "l2")}
        ttk.Label(card, text=title, style=f"{kind}.CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=vars_["big"], style="Big.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=vars_["l1"], style="CardSub.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=vars_["l2"], style="CardSub.TLabel").pack(anchor="w")
        return vars_

    def _build_playback(self):
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x", padx=20, pady=(4, 2))

        self.play_btn = ttk.Button(bar, text="▶  Animar", width=12, command=self.toggle_play)
        self.play_btn.pack(side="left")

        self.time_label = ttk.Label(bar, text="--:--", style="Time.TLabel", width=6)
        self.time_label.pack(side="left", padx=(14, 8))

        self.slider = ttk.Scale(bar, from_=0, to=1, orient="horizontal", command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True)

        ttk.Label(bar, text="Velocidade").pack(side="left", padx=(14, 5))
        self.speed_var = tk.StringVar(value="Normal")
        ttk.Combobox(bar, textvariable=self.speed_var, values=list(SPEEDS), width=8,
                     state="readonly").pack(side="left")

    def _build_figure(self):
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=20, pady=(0, 2))

        self.figure = Figure(figsize=(12, 4.8), dpi=100, facecolor=BG, layout="constrained")
        grid = self.figure.add_gridspec(2, 2, height_ratios=[1.45, 1])
        self.scene_fixed = PanelScene(self.figure.add_subplot(grid[0, 0]),
                                      "CONVENCIONAL — ângulo fixo", CONV_COLOR, axis="EW")
        self.scene_giro = PanelScene(self.figure.add_subplot(grid[0, 1]),
                                     "GIROSOL — acompanhando o Sol", GIRO_COLOR, axis="NS")
        self.chart = PowerChart(self.figure.add_subplot(grid[1, :]))

        self.canvas = FigureCanvasTkAgg(self.figure, master=container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------- simulação ----------

    def _read_inputs(self):
        try:
            day = datetime.strptime(self.date_var.get().strip(), "%d/%m/%Y").date()
        except ValueError:
            raise ValueError("Data inválida. Use o formato dd/mm/aaaa (ex.: 15/10/2026).")

        try:
            area = float(self.area_var.get().replace(",", "."))
        except ValueError:
            raise ValueError("Área inválida. Digite um número, por exemplo 1,00.")
        if area <= 0:
            raise ValueError("A área do painel precisa ser maior que zero.")

        try:
            efficiency = float(self.eff_var.get().replace(",", "."))
        except ValueError:
            raise ValueError("Eficiência inválida. Digite um número, por exemplo 20.")
        if not 0 < efficiency <= 100:
            raise ValueError("A eficiência precisa estar entre 0 e 100%.")

        start, end = int(self.start_var.get()), int(self.end_var.get())
        if end <= start:
            raise ValueError("O horário final precisa ser depois do inicial.")

        return day, start, end, area, efficiency / 100

    def run_simulation(self):
        try:
            day, start, end, area, efficiency = self._read_inputs()
        except ValueError as exc:
            messagebox.showwarning("Confira os campos", str(exc))
            return

        try:
            self.df = simulate(day, start, end, area, efficiency)
        except Exception as exc:
            messagebox.showerror("Erro na simulação",
                                 f"Não foi possível executar a simulação.\n\n{exc}")
            return

        self.pause()
        self.slider.configure(to=len(self.df) - 1)
        self.chart.plot(self.df)
        self._goto(0)
        self._update_summary()

    def _update_summary(self):
        fixed_total = self.df["fixed_energy"].iloc[-1]
        giro_total = self.df["giro_energy"].iloc[-1]
        gain = (giro_total / fixed_total - 1) * 100 if fixed_total > 0 else 0.0

        self.fixed_result.config(text=f"Convencional: {br(fixed_total)} Wh")
        self.giro_result.config(text=f"GiroSol: {br(giro_total)} Wh")
        self.diff_result.config(text=f"Ganho do GiroSol: {gain:+.1f}%".replace(".", ","))

    # ---------- quadros ----------

    def _goto(self, idx: int):
        """Move slider e cena para o ponto `idx`."""
        idx = int(np.clip(idx, 0, len(self.df) - 1))
        self._slider_lock = True
        self.slider.set(idx)
        self._slider_lock = False
        self.set_frame(idx)

    def set_frame(self, idx: int):
        if self.df is None:
            return
        self.frame_idx = idx
        r = self.df.iloc[idx]
        sun_up = r["sun_elevation"] > 0

        self.scene_fixed.update(r["sun_elevation"], r["sun_azimuth"],
                                r["fixed_tilt"], r["fixed_azimuth"], r["fixed_aoi"])
        self.scene_giro.update(r["sun_elevation"], r["sun_azimuth"],
                               r["giro_tilt"], r["giro_azimuth"], r["giro_aoi"])
        self.chart.move_cursor(r["hour"], r["fixed_power"], r["giro_power"])
        self.canvas.draw_idle()

        self.time_label.config(text=r["time"].strftime("%H:%M"))

        theta_fixed = f"{br(r['fixed_aoi'])}°" if sun_up else "—"
        theta_giro = f"{br(r['giro_aoi'])}°" if sun_up else "—"
        side = "Leste" if r["giro_azimuth"] < 180 else "Oeste"

        self.card_fixed["big"].set(f"{br(r['fixed_power'])} W")
        self.card_fixed["l1"].set(f"Energia acumulada: {br(r['fixed_energy'])} Wh")
        self.card_fixed["l2"].set(f"θ: {theta_fixed}  •  fixo a {br(r['fixed_tilt'])}° p/ Norte")

        self.card_giro["big"].set(f"{br(r['giro_power'])} W")
        self.card_giro["l1"].set(f"Energia acumulada: {br(r['giro_energy'])} Wh")
        self.card_giro["l2"].set(
            f"θ: {theta_giro}  •  giro de {br(r['giro_tilt'], 0)}° p/ {side}"
            if sun_up else "θ: —  •  painel na horizontal")

        self.card_sun["big"].set(f"{br(r['ghi'], 0)} W/m²")
        self.card_sun["l1"].set(f"Direta (DNI): {br(r['dni'], 0)}  •  Difusa (DHI): {br(r['dhi'], 0)} W/m²")
        self.card_sun["l2"].set(
            f"Elevação: {br(r['sun_elevation'])}°  •  Azimute: {br(r['sun_azimuth'], 0)}°"
            if sun_up else "Sol abaixo do horizonte")

    def _on_slider(self, value: str):
        if self._slider_lock or self.df is None:
            return
        self.pause()
        self.set_frame(int(float(value)))

    # ---------- animação ----------

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if self.df is None:
            return
        if self.frame_idx >= len(self.df) - 1:
            self._goto(0)
        self.playing = True
        self.play_btn.config(text="⏸  Pausar")
        self._tick()

    def pause(self):
        self.playing = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.play_btn.config(text="▶  Animar")

    def _tick(self):
        if not self.playing or self.df is None:
            return
        step, delay = SPEEDS[self.speed_var.get()]
        nxt = min(self.frame_idx + step, len(self.df) - 1)
        self._goto(nxt)
        if nxt >= len(self.df) - 1:
            self.pause()
            return
        self._after_id = self.root.after(delay, self._tick)


    # ---------- tabelas ----------

    def show_tables(self):
        if self.df is None:
            return
        if getattr(self, "_table_win", None) is not None and self._table_win.winfo_exists():
            self._table_win.destroy()

        df = self.df
        win = tk.Toplevel(self.root)
        self._table_win = win
        win.title("GiroSol — Tabelas")
        win.configure(bg=BG)
        sh = self.root.winfo_screenheight()
        win.geometry(f"1000x{min(680, sh - 100)}")
        win.minsize(760, 480)

        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=(self.font_family, 10))
        style.configure("Treeview.Heading", font=(self.font_family, 10, "bold"),
                        background=NAVY, foreground="white")
        style.map("Treeview.Heading", background=[("active", NAVY)])

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=14, pady=14)

        # ----- Aba 1: geração de energia -----
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="  Geração de energia  ")

        day = df["time"].iloc[0].strftime("%d/%m/%Y")
        ttk.Label(tab, text=f"Energia gerada por hora — Pelotas/RS, {day}",
                  font=(self.font_family, 12, "bold"), foreground=NAVY).pack(anchor="w")
        ttk.Label(tab, text="Painel de área e eficiência informadas; céu claro (modelo).",
                  style="Note.TLabel").pack(anchor="w", pady=(0, 8))

        cols = ("faixa", "ghi", "fixo", "giro", "dif", "ganho")
        heads = ("Faixa de horário", "Irradiância média (W/m²)", "Convencional (Wh)",
                 "GiroSol (Wh)", "Diferença (Wh)", "Ganho do GiroSol")
        widths = (150, 210, 140, 120, 120, 130)
        tree = ttk.Treeview(tab, columns=cols, show="headings", height=12)
        for c, h, w in zip(cols, heads, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor="center")
        tree.tag_configure("odd", background="#F8FAFC")
        tree.tag_configure("total", background="#DBEAFE", font=(self.font_family, 10, "bold"))

        def gain_text(fixed, giro):
            return f"{(giro / fixed - 1) * 100:+.1f}%".replace(".", ",") if fixed > 0 else "—"

        rows = hourly_table(df)
        for i, (faixa, ghi, fixed, giro) in enumerate(rows):
            tree.insert("", "end", tags=("odd",) if i % 2 else (), values=(
                faixa, br(ghi, 0), br(fixed), br(giro), br(giro - fixed, 1), gain_text(fixed, giro)))
        total_fixed, total_giro = df["fixed_energy"].iloc[-1], df["giro_energy"].iloc[-1]
        tree.insert("", "end", tags=("total",), values=(
            "TOTAL", "—", br(total_fixed), br(total_giro),
            br(total_giro - total_fixed), gain_text(total_fixed, total_giro)))
        tree.pack(fill="both", expand=True)

        peak_f = df.loc[df["fixed_power"].idxmax()]
        peak_g = df.loc[df["giro_power"].idxmax()]
        ttk.Label(
            tab, style="Note.TLabel",
            text=(f"Pico do convencional: {br(peak_f['fixed_power'])} W às {peak_f['time']:%H:%M}   •   "
                  f"Pico do GiroSol: {br(peak_g['giro_power'])} W às {peak_g['time']:%H:%M}")
        ).pack(anchor="w", pady=(8, 0))

        # ----- Aba 2: comparativo respondido -----
        tab2 = ttk.Frame(notebook, padding=12)
        notebook.add(tab2, text="  Painel fixo × Painel móvel  ")
        tab2.columnconfigure(0, weight=2, uniform="c")
        tab2.columnconfigure(1, weight=5, uniform="c")
        tab2.columnconfigure(2, weight=5, uniform="c")

        gain = (total_giro / total_fixed - 1) * 100 if total_fixed > 0 else None
        font, bold = (self.font_family, 9), (self.font_family, 10, "bold")

        for col, text in enumerate(("Aspecto", "Painel Solar Fixo", "Painel Solar Móvel")):
            tk.Label(tab2, text=text, bg=NAVY, fg="white", font=bold, pady=7).grid(
                row=0, column=col, sticky="nsew", padx=1, pady=1)
        for r, (aspect, fixed, mobile) in enumerate(comparison_rows(gain), start=1):
            bg = "white" if r % 2 else "#F1F5F9"
            tk.Label(tab2, text=aspect, bg=bg, fg=NAVY, font=bold, anchor="w",
                     justify="left", wraplength=150, padx=8, pady=6).grid(
                row=r, column=0, sticky="nsew", padx=1, pady=1)
            for col, (txt, color) in enumerate(((fixed, CONV_COLOR), (mobile, GIRO_COLOR)), start=1):
                tk.Label(tab2, text=txt, bg=bg, fg=TEXT, font=font, anchor="w", justify="left",
                         wraplength=340, padx=8, pady=8).grid(
                    row=r, column=col, sticky="nsew", padx=1, pady=1)
            tab2.rowconfigure(r, weight=1)

    # ---------- exportação e saída ----------

    def export_csv(self):
        if self.df is None:
            return
        path = filedialog.asksaveasfilename(
            title="Salvar resultados", defaultextension=".csv",
            initialfile="girosol_resultados.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return

        d = self.df
        table = pd.DataFrame({
            "Horário": d["time"].dt.strftime("%H:%M"),
            "Elevação solar (°)": d["sun_elevation"],
            "GHI (W/m²)": d["ghi"],
            "DNI (W/m²)": d["dni"],
            "DHI (W/m²)": d["dhi"],
            "Inclinação GiroSol (°)": d["giro_tilt"],
            "Incidência convencional (°)": d["fixed_aoi"],
            "Incidência GiroSol (°)": d["giro_aoi"],
            "Potência convencional (W)": d["fixed_power"],
            "Potência GiroSol (W)": d["giro_power"],
            "Energia convencional (Wh)": d["fixed_energy"],
            "Energia GiroSol (Wh)": d["giro_energy"],
        }).round(2)
        try:
            table.to_csv(path, sep=";", decimal=",", index=False, encoding="utf-8-sig")
        except OSError as exc:
            messagebox.showerror("Erro ao salvar", str(exc))

    def _on_close(self):
        self.pause()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    GiroSolApp(root)
    root.mainloop()