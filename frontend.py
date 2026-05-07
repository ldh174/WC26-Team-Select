import customtkinter as ctk
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

COUNTRY_MAP = {
    "Argentina": "ARG", "France": "FRA", "England": "ENG", "Spain": "ESP",
    "Brazil": "BRA", "Portugal": "POR", "Belgium": "BEL", "Netherlands": "NED",
    "Germany": "GER", "Colombia": "COL", "Morocco": "MAR", "United States": "USA",
    "Japan": "JPN", "Croatia": "CRO", "Senegal": "SEN"
}

COUNTRY_COLORS = {
    "Argentina": "#74ACDF", "France":        "#002395", "England":       "#CF111A",
    "Spain":     "#AA151B", "Brazil":        "#009C3B", "Portugal":      "#006600",
    "Belgium":   "#000000", "Netherlands":   "#FF6600", "Germany":       "#FFCE00",
    "Colombia":  "#FCD116", "Morocco":       "#C1272D", "United States": "#3C3B6E",
    "Japan":     "#BC002D", "Croatia":       "#FF0000", "Senegal":       "#00853F",
    "Proj. TOTT": "#FFD700"
}

POSITION_AXES = {
    "FW": ("sh_p90",      "gls_p90"),
    "DF": ("tkl_won_adj", "interceptions_adj"),
    "GK": ("gk_save_pct", "gk_cs"),
    "MF": ("ast_p90",     "interceptions_adj"),
}

AXIS_LABELS = {
    "sh_p90":            "Shots per 90",
    "gls_p90":           "Goals per 90",
    "tkl_won_adj":       "Tackles Won (Adj.)",
    "interceptions_adj": "Interceptions (Adj.)",
    "gk_save_pct":       "Save %",
    "gk_cs":             "Clean Sheets",
    "ast_p90":           "Assists per 90",
}

STAT_GROUPS = {
    "Attack": [
        ("Goals",           "gls"),
        ("Assists",         "ast"),
        ("Goals per 90",    "gls_p90"),
        ("Assists per 90",  "ast_p90"),
        ("Shots",           "sh"),
        ("Shots on Target", "sot"),
        ("Shot Acc. %",     "sot_pct"),
        ("Non-PK Goals",    "gls_npk"),
        ("G+A",             "g_a"),
    ],
    "Possession": [
        ("Crosses",         "crosses"),
        ("G+A per 90",      "g_a_p90"),
        ("Shots per 90",    "sh_p90"),
        ("SoT per 90",      "sot_p90"),
    ],
    "Defence": [
        ("Tackles Won",     "tkl_won"),
        ("Interceptions",   "interceptions"),
        ("Tkl Won (Adj.)",  "tkl_won_adj"),
        ("Int. (Adj.)",     "interceptions_adj"),
    ],
    "Goalkeeping": [
        ("Goals Against",       "gk_ga"),
        ("GA per 90",           "gk_ga90"),
        ("Shots on Target Fcd.","gk_sota"),
        ("Saves",               "gk_saves"),
        ("Save %",              "gk_save_pct"),
        ("Clean Sheets",        "gk_cs"),
        ("CS %",                "gk_cs_pct"),
        ("PKs Faced",           "gk_pk_faced"),
        ("PKs Allowed",         "gk_pk_allowed"),
        ("PKs Saved",           "gk_pk_saved"),
        ("PK Save %",           "gk_pk_save_pct"),
    ],
    "Physical": [
        ("Minutes",         "min"),
        ("Appearances",     "mp"),
        ("Starts",          "starts"),
        ("90s Played",      "90s"),
    ],
    "Discipline": [
        ("Yellow Cards",    "yellow"),
        ("Red Cards",       "red"),
        ("Fouls",           "fouls"),
        ("Fouled",          "fouled"),
        ("Offsides",        "offsides"),
    ],
}

CORE_STATS = [
    ("Weighted Score",  "weighted_score",   lambda v: f"{v:.3f}"),
    ("Age",             "age",              lambda v: str(int(v)) if pd.notna(v) else "-"),
    ("Position",        "position_group",   str),
    ("Appearances",     "mp",               lambda v: str(int(v)) if pd.notna(v) else "-"),
    ("Minutes Played",  "min",              lambda v: str(int(v)) if pd.notna(v) else "-"),
]

COUNTRIES     = list(COUNTRY_MAP.keys())
VIZ_COUNTRIES = COUNTRIES + ["Proj. TOTT"]
POSITIONS     = ["GK", "DF", "MF", "FW"]
CSV_DIR       = "squads"
MIN_SUBS      = {"GK": 2, "DF": 2, "MF": 2, "FW": 2}


def fmt(val, formatter):
    try:
        return formatter(val)
    except Exception:
        return "-"


def load_squad(country_name):
    filename = f"{CSV_DIR}/{country_name.lower().replace(' ', '_')}_squad.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame()


class TitleScreen(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.df = pd.read_csv(f"{CSV_DIR}/squads_combined.csv")

        tott_path = f"{CSV_DIR}/tott.csv"
        if os.path.exists(tott_path):
            tott_df = pd.read_csv(tott_path)
            tott_df["_tott"] = True
            self.tott_df = tott_df
        else:
            self.tott_df = pd.DataFrame()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.titleLabel = ctk.CTkLabel(self, text="World Cup Team Selector", font=("calibri", 50), text_color="#0077BE")
        self.titleLabel.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.tabview = ctk.CTkTabview(self)
        self.tabview._segmented_button.configure(font=("calibri", 25))
        self.tabview.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        for tab in ("Methodology", "Rankings", "Team", "Visualization", "Head to Head"):
            self.tabview.add(tab)

        methodology_text = (
            "    From the 2023-24 season stretching to the end of the March international break of the "
            "2025-26 season, we first scraped a great amount of raw player data from the top 11 leagues "
            "and then proceeded to clean it (removing redundant information, inapplicable data, etc.). "
            "After normalizing and cleaning the data, we developed an algorithm to output a weighted score "
            "that considers all sorts of facts about the players like their game time, fitness record, "
            "discipline record, and stats according to their position specifically.\n\n"
            "    Frequent reiterations was done to the model to account for scenarios that data cannot "
            "predict: retirees were excluded, players who play a different position for the national team "
            "were considered, and players with lackluster club careers but great records with their national "
            "team were given special weighting.\n\n"
            "    We have provided several tools to aid with your national team selection: a ranking tab will "
            "tell, descending, who has the highest weighted score. The Team tab will tell you our model's "
            "starting 11, substitutes, and alternates that each national team manager should pursue. "
            "Additionally, we have provided a data visualization and head-to-head player comparison tab "
            "that will aid with further analysis.\n\n"
            "    By Laurent Drejaj and Reda Abdel-Aziz."
        )
        method_tab = self.tabview.tab("Methodology")
        method_tab.grid_columnconfigure(0, weight=1)
        method_tab.grid_rowconfigure(0, weight=1)
        method_scroll = ctk.CTkScrollableFrame(method_tab)
        method_scroll.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        method_scroll.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            method_scroll,
            text=methodology_text,
            font=("calibri", 16),
            wraplength=900,
            justify="left"
        ).grid(row=0, column=0, pady=20, padx=20, sticky="w")

        self.build_rankings_tab()
        self.build_team_tab()
        self.build_visualization_tab()
        self.build_h2h_tab()

    def build_rankings_tab(self):
        tab = self.tabview.tab("Rankings")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        country_frame = ctk.CTkScrollableFrame(tab, orientation="horizontal", height=55)
        country_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))

        ctk.CTkLabel(country_frame, text="Countries:", font=("calibri", 13, "bold")).pack(side="left", padx=(5, 8))
        self.country_vars = {}
        for name in COUNTRIES:
            var = ctk.BooleanVar(value=False)
            self.country_vars[name] = var
            ctk.CTkCheckBox(
                country_frame, text=name, variable=var,
                font=("calibri", 12), command=self.apply_filters
            ).pack(side="left", padx=4)

        pos_frame = ctk.CTkFrame(tab, height=45)
        pos_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 8))

        ctk.CTkLabel(pos_frame, text="Position:", font=("calibri", 13, "bold")).pack(side="left", padx=(10, 8))
        self.pos_vars = {}
        for pos in POSITIONS:
            var = ctk.BooleanVar(value=False)
            self.pos_vars[pos] = var
            ctk.CTkCheckBox(
                pos_frame, text=pos, variable=var,
                font=("calibri", 12), command=self.apply_filters
            ).pack(side="left", padx=10)

        list_frame = ctk.CTkFrame(tab)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        list_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        list_frame.grid_rowconfigure(1, weight=1)

        for col, text in enumerate(["Rank", "Player", "Nation", "Position", "Age", "Weighted Score"]):
            ctk.CTkLabel(
                list_frame, text=text,
                font=("calibri", 14, "bold"), text_color="#0077BE"
            ).grid(row=0, column=col, padx=10, pady=(8, 4), sticky="w")

        self.results_frame = ctk.CTkScrollableFrame(list_frame)
        self.results_frame.grid(row=1, column=0, columnspan=6, sticky="nsew", padx=5, pady=5)
        for col in range(6):
            self.results_frame.grid_columnconfigure(col, weight=1)

        self.apply_filters()

    def apply_filters(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        selected_codes = {COUNTRY_MAP[n] for n, v in self.country_vars.items() if v.get()}
        selected_pos   = {pos for pos, v in self.pos_vars.items() if v.get()}
        if not selected_pos:
            selected_pos = set(POSITIONS)

        filtered = self.df[
            self.df["nation"].isin(selected_codes) &
            self.df["position_group"].isin(selected_pos)
        ].sort_values("weighted_score", ascending=False).reset_index(drop=True)

        for i, row in filtered.iterrows():
            rank = i + 1
            bg = "#3a3a3a" if rank % 2 == 0 else "#676767"
            for col, val in enumerate([
                rank, row["player"], row["nation"], row["position_group"],
                int(row["age"]) if pd.notna(row["age"]) else "-",
                f'{row["weighted_score"]:.3f}'
            ]):
                ctk.CTkLabel(
                    self.results_frame, text=str(val),
                    font=("calibri", 13), fg_color=bg,
                    corner_radius=4, text_color="white"
                ).grid(row=rank, column=col, padx=6, pady=2, sticky="ew")

    def build_team_tab(self):
            tab = self.tabview.tab("Team")
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(1, weight=1)

            top_frame = ctk.CTkFrame(tab)
            top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
            top_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(top_frame, text="Country:", font=("calibri", 15, "bold")).grid(row=0, column=0, padx=(10, 8), pady=8)
            self.country_dropdown = ctk.CTkOptionMenu(
                top_frame, values=COUNTRIES + ["Proj. TOTT"],
                font=("calibri", 14), command=self.load_team
            )
            self.country_dropdown.grid(row=0, column=1, padx=8, pady=8, sticky="w")

            self.team_scroll = ctk.CTkScrollableFrame(tab)
            self.team_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
            self.team_scroll.grid_columnconfigure(0, weight=1)

            self.load_team(COUNTRIES[0])

    def load_team(self, country_name):
        for widget in self.team_scroll.winfo_children():
            widget.destroy()

        if country_name == "Proj. TOTT":
            if self.tott_df.empty:
                ctk.CTkLabel(self.team_scroll, text="No TOTT data found (tott.csv missing).",
                             font=("calibri", 14), text_color="red").pack(pady=20)
                return
            df = self.tott_df.copy()
        else:
            filename = f"{CSV_DIR}/{country_name.lower().replace(' ', '_')}_squad.csv"
            if not os.path.exists(filename):
                ctk.CTkLabel(self.team_scroll, text=f"No data found for {country_name}.",
                             font=("calibri", 14), text_color="red").pack(pady=20)
                return
            df = pd.read_csv(filename)

        STARTER_COUNTS = {"GK": 1, "DF": 4, "MF": 3, "FW": 3}
        starters = {pos: df[df["position_group"] == pos]
                        .sort_values("weighted_score", ascending=False)
                        .head(n)
                    for pos, n in STARTER_COUNTS.items()}
        starter_idx   = pd.concat(starters.values()).index
        subs_df       = df[~df.index.isin(starter_idx) & df["role"].isin(["starter", "backup", "sub"])].sort_values("weighted_score", ascending=False)
        alternates_df = df[df["role"] == "alternate"].sort_values("weighted_score", ascending=False)

        if country_name == "Proj. TOTT":
            ctk.CTkLabel(
                self.team_scroll, text="Projected Team of the Tournament",
                font=("calibri", 16, "bold"), text_color="#FFD700"
            ).pack(pady=(6, 2))

        self._render_formation(starters)
        self._render_section("SUBS", subs_df)
        self._render_section("ALTERNATES", alternates_df)

    def _player_label(self, parent, name):
        ctk.CTkLabel(
            parent, text=name,
            font=("calibri", 13, "bold"),
            fg_color="#0077BE", text_color="white",
            corner_radius=6, padx=8, pady=4
        ).pack(side="left", padx=6, pady=4)

    def _formation_row(self, players):
        row_frame = ctk.CTkFrame(self.team_scroll, fg_color="transparent")
        row_frame.pack(fill="x", pady=4)
        inner = ctk.CTkFrame(row_frame, fg_color="transparent")
        inner.pack(anchor="center")
        for _, player in players.iterrows():
            self._player_label(inner, player["player"])

    def _render_formation(self, starters):
        ctk.CTkLabel(
            self.team_scroll, text="── STARTING XI ──",
            font=("calibri", 15, "bold"), text_color="#0077BE"
        ).pack(pady=(10, 4))
        for pos in ["FW", "MF", "DF", "GK"]:
            self._formation_row(starters[pos])

    def _render_section(self, title, players_df):
        ctk.CTkLabel(
            self.team_scroll, text=f"── {title} ──",
            font=("calibri", 15, "bold"), text_color="#0077BE"
        ).pack(pady=(14, 4))
        for pos in ["GK", "DF", "MF", "FW"]:
            group = players_df[players_df["position_group"] == pos]
            if group.empty:
                continue
            row_frame = ctk.CTkFrame(self.team_scroll, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(row_frame, text=f"{pos}:", font=("calibri", 12, "bold"),
                         text_color="gray", width=40).pack(side="left", padx=(10, 4))
            for _, player in group.iterrows():
                self._player_label(row_frame, player["player"])

    def build_visualization_tab(self):
        tab = self.tabview.tab("Visualization")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(controls, text="Teams:", font=("calibri", 13, "bold")).pack(side="left", padx=(10, 4))

        self.viz_all_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            controls, text="All", variable=self.viz_all_var,
            font=("calibri", 12, "bold"), text_color="#0077BE",
            command=self._viz_toggle_all
        ).pack(side="left", padx=(0, 8))

        self.viz_country_scroll = ctk.CTkScrollableFrame(controls, orientation="horizontal", height=45, width=700)
        self.viz_country_scroll.pack(side="left", fill="x", expand=True)

        self.viz_country_vars = {}
        for name in VIZ_COUNTRIES:
            var = ctk.BooleanVar(value=False)
            self.viz_country_vars[name] = var
            color = "#FFD700" if name == "Proj. TOTT" else "black"
            ctk.CTkCheckBox(
                self.viz_country_scroll, text=name, variable=var,
                font=("calibri", 12), text_color=color,
                command=self._viz_country_changed
            ).pack(side="left", padx=4)

        ctk.CTkLabel(controls, text="  Position:", font=("calibri", 13, "bold")).pack(side="left", padx=(10, 4))
        self.viz_pos_var = ctk.StringVar(value="FW")
        ctk.CTkOptionMenu(
            controls, values=POSITIONS,
            variable=self.viz_pos_var,
            font=("calibri", 13),
            command=lambda _: self.update_viz()
        ).pack(side="left", padx=(0, 10))

        self.viz_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.viz_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)

        self.viz_canvas = None
        self.update_viz()

    def _viz_toggle_all(self):
        state = self.viz_all_var.get()
        for var in self.viz_country_vars.values():
            var.set(state)
        self.update_viz()

    def _viz_country_changed(self):
        all_checked = all(v.get() for v in self.viz_country_vars.values())
        self.viz_all_var.set(all_checked)
        self.update_viz()

    def update_viz(self):
        selected_countries = [n for n, v in self.viz_country_vars.items() if v.get()]
        pos           = self.viz_pos_var.get()
        x_col, y_col  = POSITION_AXES[pos]

        if self.viz_canvas:
            self.viz_canvas.get_tk_widget().destroy()
            self.viz_canvas = None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#2b2b2b")
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.grid(color="#3a3a3a", linestyle="--", linewidth=0.5)

        if not selected_countries:
            ax.text(0.5, 0.5, "Select at least one team to display",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=14)
        else:
            legend_patches = []
            for country in selected_countries:
                if country == "Proj. TOTT":
                    subset = self.tott_df
                    if subset.empty:
                        continue
                    subset = subset[subset["position_group"] == pos].dropna(subset=[x_col, y_col, "weighted_score"])
                else:
                    code   = COUNTRY_MAP[country]
                    subset = self.df[
                        (self.df["nation"] == code) &
                        (self.df["position_group"] == pos)
                    ].dropna(subset=[x_col, y_col, "weighted_score"])

                if subset.empty:
                    continue

                color = COUNTRY_COLORS[country]
                sizes = ((subset["weighted_score"] - subset["weighted_score"].min() + 1) * 30).clip(lower=40)

                ax.scatter(
                    subset[x_col], subset[y_col],
                    s=sizes, c=color, alpha=0.85,
                    edgecolors="white", linewidths=0.4
                )
                for _, row in subset.iterrows():
                    ax.annotate(
                        row["player"], (row[x_col], row[y_col]),
                        fontsize=6.5, color="white", alpha=0.85,
                        xytext=(4, 4), textcoords="offset points"
                    )
                legend_patches.append(mpatches.Patch(color=color, label=country))

            if legend_patches:
                ax.legend(
                    handles=legend_patches, loc="upper left",
                    facecolor="#2b2b2b", edgecolor="#555555",
                    labelcolor="white", fontsize=9
                )

        ax.set_xlabel(AXIS_LABELS.get(x_col, x_col), fontsize=11)
        ax.set_ylabel(AXIS_LABELS.get(y_col, y_col), fontsize=11)
        ax.set_title(f"{pos} — {AXIS_LABELS.get(x_col, x_col)} vs {AXIS_LABELS.get(y_col, y_col)}", fontsize=13)
        fig.tight_layout()

        self.viz_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        plt.close(fig)

    def build_h2h_tab(self):
        tab = self.tabview.tab("Head to Head")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        selectors = ctk.CTkFrame(tab)
        selectors.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        selectors.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(selectors, text="Team 1:", font=("calibri", 13, "bold")).grid(row=0, column=0, padx=(10, 4), pady=8, sticky="w")
        self.h2h_team1 = ctk.CTkOptionMenu(
            selectors, values=COUNTRIES, font=("calibri", 13),
            command=lambda v: self._h2h_update_players(1, v)
        )
        self.h2h_team1.grid(row=0, column=1, padx=4, pady=8, sticky="ew")
        self.h2h_team1.set("Argentina")

        ctk.CTkLabel(selectors, text="Player 1:", font=("calibri", 13, "bold")).grid(row=0, column=2, padx=(16, 4), pady=8, sticky="w")
        self.h2h_player1 = ctk.CTkOptionMenu(
            selectors, values=["—"], font=("calibri", 13),
            command=lambda _: self._h2h_render()
        )
        self.h2h_player1.grid(row=0, column=3, padx=4, pady=8, sticky="ew")

        ctk.CTkLabel(selectors, text="Team 2:", font=("calibri", 13, "bold")).grid(row=1, column=0, padx=(10, 4), pady=8, sticky="w")
        self.h2h_team2 = ctk.CTkOptionMenu(
            selectors, values=COUNTRIES, font=("calibri", 13),
            command=lambda v: self._h2h_update_players(2, v)
        )
        self.h2h_team2.grid(row=1, column=1, padx=4, pady=8, sticky="ew")
        self.h2h_team2.set("France")

        ctk.CTkLabel(selectors, text="Player 2:", font=("calibri", 13, "bold")).grid(row=1, column=2, padx=(16, 4), pady=8, sticky="w")
        self.h2h_player2 = ctk.CTkOptionMenu(
            selectors, values=["—"], font=("calibri", 13),
            command=lambda _: self._h2h_render()
        )
        self.h2h_player2.grid(row=1, column=3, padx=4, pady=8, sticky="ew")

        self.h2h_scroll = ctk.CTkScrollableFrame(tab)
        self.h2h_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.h2h_scroll.grid_columnconfigure((0, 1, 2), weight=1)

        self.h2h_df1 = pd.DataFrame()
        self.h2h_df2 = pd.DataFrame()

        self._h2h_update_players(1, "Argentina")
        self._h2h_update_players(2, "France")

    def _h2h_update_players(self, slot, country_name):
        df      = load_squad(country_name)
        players = sorted(df["player"].dropna().tolist()) if not df.empty else ["—"]
        if slot == 1:
            self.h2h_df1 = df
            self.h2h_player1.configure(values=players)
            self.h2h_player1.set(players[0])
        else:
            self.h2h_df2 = df
            self.h2h_player2.configure(values=players)
            self.h2h_player2.set(players[0])
        self._h2h_render()

    def _h2h_render(self):
        for widget in self.h2h_scroll.winfo_children():
            widget.destroy()

        n1 = self.h2h_player1.get()
        n2 = self.h2h_player2.get()

        if self.h2h_df1.empty or self.h2h_df2.empty or n1 == "—" or n2 == "—":
            ctk.CTkLabel(self.h2h_scroll, text="Select two players to compare.",
                         font=("calibri", 14), text_color="gray").pack(pady=30)
            return

        r1 = self.h2h_df1[self.h2h_df1["player"] == n1].iloc[0]
        r2 = self.h2h_df2[self.h2h_df2["player"] == n2].iloc[0]

        ctk.CTkLabel(
            self.h2h_scroll,
            text=f"{n1}  vs  {n2}",
            font=("calibri", 20, "bold"), text_color="#0077BE"
        ).pack(pady=(10, 6))

        self._h2h_stat_rows(CORE_STATS, r1, r2)

        for group_name, stats in STAT_GROUPS.items():
            self._h2h_collapsible(group_name, stats, r1, r2)

    def _h2h_stat_rows(self, stats, r1, r2):
        frame = ctk.CTkFrame(self.h2h_scroll, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=2)
        frame.grid_columnconfigure((0, 1, 2), weight=1)

        for i, (label, col, formatter) in enumerate(stats):
            v1 = fmt(r1.get(col, "-"), formatter)
            v2 = fmt(r2.get(col, "-"), formatter)
            bg = "#3a3a3a" if i % 2 == 0 else "#2b2b2b"

            row_frame = ctk.CTkFrame(frame, fg_color=bg, corner_radius=4)
            row_frame.pack(fill="x", pady=1)
            row_frame.grid_columnconfigure((0, 1, 2), weight=1)

            ctk.CTkLabel(row_frame, text=v1, font=("calibri", 13),
                         text_color="white").grid(row=0, column=0, padx=10, pady=5, sticky="e")
            ctk.CTkLabel(row_frame, text=label, font=("calibri", 12, "bold"),
                         text_color="#0077BE").grid(row=0, column=1, padx=10, pady=5)
            ctk.CTkLabel(row_frame, text=v2, font=("calibri", 13),
                         text_color="white").grid(row=0, column=2, padx=10, pady=5, sticky="w")

    def _h2h_collapsible(self, title, stats, r1, r2):
        wrapper = ctk.CTkFrame(self.h2h_scroll, fg_color="#1e1e1e", corner_radius=6)
        wrapper.pack(fill="x", padx=10, pady=4)

        content_frame = ctk.CTkFrame(wrapper, fg_color="transparent")
        is_open       = ctk.BooleanVar(value=False)

        def toggle():
            if is_open.get():
                content_frame.pack_forget()
                is_open.set(False)
                header_btn.configure(text=f"▶  {title}")
            else:
                content_frame.pack(fill="x", padx=4, pady=(0, 6))
                is_open.set(True)
                header_btn.configure(text=f"▼  {title}")

        header_btn = ctk.CTkButton(
            wrapper, text=f"▶  {title}",
            font=("calibri", 13, "bold"),
            fg_color="transparent", text_color="#0077BE",
            hover_color="#2a2a2a", anchor="w",
            command=toggle
        )
        header_btn.pack(fill="x", padx=6, pady=4)

        fmt_default          = lambda v: f"{float(v):.2f}" if pd.notna(v) else "-"
        stat_triples         = [(lbl, col, fmt_default) for lbl, col in stats]
        content_frame_inner  = ctk.CTkFrame(content_frame, fg_color="transparent")
        content_frame_inner.pack(fill="x")

        for i, (label, col, formatter) in enumerate(stat_triples):
            v1 = fmt(r1.get(col, "-"), formatter)
            v2 = fmt(r2.get(col, "-"), formatter)
            bg = "#3a3a3a" if i % 2 == 0 else "#2b2b2b"

            row_frame = ctk.CTkFrame(content_frame_inner, fg_color=bg, corner_radius=4)
            row_frame.pack(fill="x", pady=1)
            row_frame.grid_columnconfigure((0, 1, 2), weight=1)

            ctk.CTkLabel(row_frame, text=v1, font=("calibri", 13),
                         text_color="white").grid(row=0, column=0, padx=10, pady=4, sticky="e")
            ctk.CTkLabel(row_frame, text=label, font=("calibri", 12, "bold"),
                         text_color="#0077BE").grid(row=0, column=1, padx=10, pady=4)
            ctk.CTkLabel(row_frame, text=v2, font=("calibri", 13),
                         text_color="white").grid(row=0, column=2, padx=10, pady=4, sticky="w")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("World Cup Squad Selector")
        self.geometry("1280x960")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.title_frame = TitleScreen(master=self)
        self.title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")


app = App()
app.mainloop()