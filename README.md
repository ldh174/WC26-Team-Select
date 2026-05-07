# WC26-Team-Select
Data-driven software for World Cup 2026 national team squad selection. Combines player statistics from 12 professional leagues across 3 seasons, applies a multi-factor weighted scoring algorithm, and generates ranked 26-man squads for 15 nations. Complete with a GUI for squad viewing, player rankings, data visualization, and head-to-head comparison. Details on how to work the system below.

---
## How It Works
Player stats are sourced from FBref across 12 leagues: EPL, Bundesliga, La Liga, Serie A, Ligue 1, Eredivisie, Belgian Pro League, Primeira Liga, Saudi Pro League, Süper Lig, MLS, and UCL. National team appearance data comes from a Kaggle football dataset covering career international caps per player.
 
Each player receives a weighted score built from several stages. First, all statistics are normalized per-90 minutes. Then they are scaled by a league difficulty coefficient so that performance in stronger leagues carries more weight. Second, a position-specific score is computed using separate formulas for GK, DF, MF, and FW that weight relevant stats accordingly. Third, scores are min-max normalized to a 0–100 scale within each position group. Fourth, up to three seasons of data are aggregated using a 50/30/20 weighting that prioritizes recent form. A form trend bonus is then applied based on season-over-season stat changes. Finally a national team caps multiplier rewards players with proven international track record.
 
Squad selection picks the top 26 players per nation (3 GK, 8 DF, 8 MF, 7 FW) with position override corrections for FBref misclassifications and reputation boosts for players whose club stats are suppressed by injury or rotation. 15 nations are covered: Argentina, France, England, Spain, Brazil, Portugal, Belgium, Netherlands, Germany, Colombia, Morocco, United States, Japan, Croatia, and Senegal.

---

## GUI
The app has five tabs. The Methodology tab explains the scoring formulas and algorithm. The Rankings tab shows player ranking by nation and position sorted by weighted score. The Team tab displays the 4-3-3 starting XI, substitutes, and alternates for each nation (a also includes a projected TOTT option showing the projected Team of the Tournament and U23 Team of the Tournament across all 15 nations is also included). The Visualization tab showcases position-specific scatter plots comparing stats across selected nations. The Head to Head tab allows side-by-side stat comparison between any two players with stat group sections.

---
# Setup
## Windows
```
git clone https://github.com/ldh174/WC26-Team-Select.git
cd WC26-Team-Select
pip install pandas numpy customtkinter matplotlib pillow
```

## macOS
```
git clone https://github.com/ldh174/WC26-Team-Select.git
cd WC26-Team-Select
pip3 install pandas numpy customtkinter matplotlib pillow
```

## Linux
```
git clone https://github.com/ldh174/WC26-Team-Select.git
cd WC26-Team-Select
pip install pandas numpy customtkinter matplotlib pillow --break-system-packages
```
---
# How to Run
All data files included no need for external downloads. The squads are pre-generated so the minimum to launch the GUI is just: `python3 frontend.py` \
To regenerate squads from scratch after modifying the algorithm:
## Windows
```
python clean.py
python feature_engineering.py
python merge_caps.py
python select_squad.py --all
python frontend.py
```

## macOS/Linux
```
python3 clean.py
python3 feature_engineering.py
python3 merge_caps.py
python3 select_squad.py --all
python3 frontend.py
```

## Extras
```
python3 select_squad.py --country "Germany"   # single nation
python3 tott.py --both                        # tott + u23 tott
```
---
*By Laurent Drejaj and Reda Abdel-Aziz*
