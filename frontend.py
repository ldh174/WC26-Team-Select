import customtkinter as ctk
import pandas as pd
import os

COUNTRY_MAP = {
    "Argentina": "ARG", "France": "FRA", "England": "ENG", "Spain": "ESP",
    "Brazil": "BRA", "Portugal": "POR", "Belgium": "BEL", "Netherlands": "NED",
    "Germany": "GER", "Colombia": "COL", "Morocco": "MAR", "United States": "USA",
    "Japan": "JPN", "Croatia": "CRO", "Senegal": "SEN"
}

COUNTRIES = list(COUNTRY_MAP.keys())
POSITIONS = ["GK", "DF", "MF", "FW"]
CSV_DIR = "WC26-Team-Select/squads"

MIN_SUBS = {"GK": 2, "DF": 2, "MF": 2, "FW": 2}


class TitleScreen(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.df = pd.read_csv(f"{CSV_DIR}/squads_combined.csv")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.titleLabel = ctk.CTkLabel(self, text="World Cup Team Selector", font=("calibri", 50), text_color="#0077BE")
        self.titleLabel.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.tabview = ctk.CTkTabview(self)
        self.tabview._segmented_button.configure(font=("calibri", 25))
        self.tabview.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        self.tabview.add("Methodology")
        self.tabview.add("Rankings")
        self.tabview.add("Team")

        methodology_text = (
            "Starting from the 2023-24 season and ending at the international break of the "
            "2025-26 season, all useful player data is compiled and weighted and then computed "
            "into a weighted score. This makes notes of other factors such as league difficulty, "
            "injury history, etc. Using this score, we output what the model believes is the best "
            "team for a manager to call up for the 2026 World Cup."
        )
        ctk.CTkLabel(
            self.tabview.tab("Methodology"),
            text=methodology_text,
            font=("calibri", 24),
            wraplength=650,
            justify="left"
        ).pack(pady=30, padx=20, anchor="w")

        self.build_rankings_tab()
        self.build_team_tab()

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

        selected_pos = {pos for pos, v in self.pos_vars.items() if v.get()}
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
                rank,
                row["player"],
                row["nation"],
                row["position_group"],
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
            top_frame, values=COUNTRIES,
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

        filename = f"{CSV_DIR}/{country_name.lower().replace(' ', '_')}_squad.csv"
        if not os.path.exists(filename):
            ctk.CTkLabel(self.team_scroll, text=f"No data found for {country_name}.",
                         font=("calibri", 14), text_color="red").pack(pady=20)
            return

        df = pd.read_csv(filename)

        def top_n(pos, n):
            return df[df["position_group"] == pos].sort_values("weighted_score", ascending=False).head(n)

        starters = {
            "FW": top_n("FW", 3),
            "MF": top_n("MF", 3),
            "DF": top_n("DF", 4),
            "GK": top_n("GK", 1),
        }
        starter_indices = pd.concat(starters.values()).index

        remaining = df[~df.index.isin(starter_indices)].copy()

        subs = []
        sub_indices = set()

        for pos, min_count in MIN_SUBS.items():
            pool = remaining[remaining["position_group"] == pos].sort_values("weighted_score", ascending=False).head(min_count)
            subs.append(pool)
            sub_indices.update(pool.index)

        already_used = starter_indices.tolist() + list(sub_indices)
        fill_pool = df[~df.index.isin(already_used)].sort_values("weighted_score", ascending=False)
        spots_left = 15 - len(sub_indices)
        if spots_left > 0:
            filler = fill_pool.head(spots_left)
            subs.append(filler)
            sub_indices.update(filler.index)

        subs_df = pd.concat(subs).sort_values("weighted_score", ascending=False)

        all_used = starter_indices.tolist() + list(sub_indices)
        alternates_df = df[~df.index.isin(all_used)].sort_values("weighted_score", ascending=False)

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


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("World Cup Squad Selector")
        self.geometry("800x600")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.title_frame = TitleScreen(master=self)
        self.title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

app = App()
app.mainloop()