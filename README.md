# FPL Squad Builder

Web app that builds a legal **15-man Fantasy Premier League squad** maximising projected points under official rules — the browser equivalent of an Excel squad optimiser.

## Rules enforced

- 2 GKP · 5 DEF · 5 MID · 3 FWD  
- Budget (default £100.0m, adjustable)  
- Max 3 players from the same club  
- Optional: lock / exclude players  

## Projection

Transparent blend of FPL `ep_next`, form and points-per-game, adjusted for availability.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)

1. Push this folder to a public GitHub repo.  
2. Go to https://share.streamlit.io → New app → select repo → main file `app.py`.  
3. Deploy.  

Embed on your site with:

```html
<iframe
  src="https://YOUR-APP.streamlit.app/?embed=true"
  width="100%"
  height="1100"
  style="border:none;border-radius:12px;"
  loading="lazy"
  allowfullscreen>
</iframe>
```

## Data

Public Fantasy Premier League API only.
