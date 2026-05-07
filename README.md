# WC26-Team-Select
Data driven software to help with World Cup 2026 national team callups.

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

# How to Run
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

## Notes
- All data files are included
- Rerun full pipeline after any changes to select_squad.py or feature_engineering.py
