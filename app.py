import streamlit as st
import pandas as pd
from io import BytesIO

from squad_engine import SquadBuilder, SquadConfig

st.set_page_config(
    page_title="FPL Squad Builder",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ FPL Squad Builder")
st.caption(
    "Build a legal 15-man Fantasy Premier League squad that maximises projected points "
    "under official budget and club limits — the same idea as an Excel optimiser, in the browser."
)

with st.expander("📖 How this works", expanded=False):
    st.markdown(
        """
**What the optimiser does**
1. Pulls the latest player prices, form and expected points from the official FPL API.
2. Builds a projected points score for every player (blend of expected points, form and season average).
3. Finds the **best valid 15-man squad** under the rules:
   - 2 GKP · 5 DEF · 5 MID · 3 FWD  
   - Budget (default £100.0m)  
   - Max 3 players from the same club  
4. Suggests a starting XI, captain and vice-captain.

**Tips**
- Lock players you already own and want to keep.
- Exclude players you refuse to touch.
- Lower the budget if you are mid-season and have less than £100m bank + squad value for a wildcard-style rebuild.
        """
    )

@st.cache_resource
def get_builder():
    try:
        return SquadBuilder().load()
    except Exception as e:
        st.error(f"Could not load FPL data: {e}")
        st.stop()


builder = get_builder()
players = builder.players

if players is None or players.empty:
    st.error("No player data available.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Squad settings")

    budget = st.slider(
        "Budget (£m)",
        min_value=80.0,
        max_value=105.0,
        value=100.0,
        step=0.5,
        help="Total squad budget. Use less than 100 if you are not on a full wildcard reset.",
    )

    max_per_team = st.slider("Max players per club", 1, 3, 3)

    min_minutes = st.slider(
        "Minimum minutes this season",
        0,
        900,
        0,
        30,
        help="Filter out players who have barely played.",
    )

    differential = st.slider(
        "Differential preference",
        0.0,
        0.3,
        0.0,
        0.05,
        help="Slightly boost lower-owned players (0 = pure points).",
    )

    st.divider()
    st.subheader("Lock / exclude players")

    name_list = (
        players.sort_values(["position", "player"])
        .apply(lambda r: f"{r['player']} ({r['team_short']}) · {r['position']} · £{r['price']:.1f}m", axis=1)
        .tolist()
    )
    id_by_label = {
        f"{r['player']} ({r['team_short']}) · {r['position']} · £{r['price']:.1f}m": int(r["id"])
        for _, r in players.iterrows()
    }

    locked_labels = st.multiselect(
        "Must include (locked)",
        options=name_list,
        help="These players will be forced into the squad.",
    )
    excluded_labels = st.multiselect(
        "Must exclude",
        options=name_list,
        help="These players will never be picked.",
    )

    locked_ids = {id_by_label[x] for x in locked_labels if x in id_by_label}
    excluded_ids = {id_by_label[x] for x in excluded_labels if x in id_by_label}

    # Remove overlap
    locked_ids -= excluded_ids

    run = st.button("🚀 Build optimal squad", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------
if run:
    cfg = SquadConfig(
        budget=budget,
        max_per_team=max_per_team,
        min_minutes=min_minutes,
        differential_boost=differential,
    )
    builder.config = cfg
    # Re-filter with new min_minutes
    builder._build_players()

    with st.spinner("Optimising squad…"):
        result = builder.optimise(
            locked_ids=locked_ids,
            excluded_ids=excluded_ids,
            budget=budget,
            max_per_team=max_per_team,
        )

    if not result.get("ok"):
        st.error(result.get("error", "Optimisation failed."))
        st.stop()

    squad = result["squad"]
    xi = result["starting_xi"]
    bench = result["bench"]

    st.success(
        f"**Squad ready** · £{result['total_cost']:.1f}m spent · "
        f"£{result['bank']:.1f}m bank · "
        f"Projected {result['total_projected']:.1f} pts (raw sum)"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Squad cost", f"£{result['total_cost']:.1f}m")
    c2.metric("Bank", f"£{result['bank']:.1f}m")
    c3.metric("Players", f"{result['player_count']}")
    c4.metric("Projected pts", f"{result['total_projected']:.1f}")

    # ---- Starting XI ----
    st.subheader("Starting XI (suggested)")
    if xi is None or xi.empty:
        st.warning("Could not build a starting XI from this squad.")
    else:
        if "role" not in xi.columns:
            xi = xi.copy()
            xi["role"] = ""
        cols = [c for c in ["player", "team_short", "position", "price", "projected", "ownership", "role"] if c in xi.columns]
        xi_view = xi[cols].copy()
        rename = {
            "player": "Player", "team_short": "Team", "position": "Pos",
            "price": "Price £m", "projected": "Projected", "ownership": "Own %", "role": "Role",
        }
        xi_view = xi_view.rename(columns=rename)
        st.dataframe(
            xi_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price £m": st.column_config.NumberColumn(format="£%.1f"),
                "Projected": st.column_config.NumberColumn(format="%.2f"),
                "Own %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    # ---- Bench ----
    st.subheader("Bench")
    if not bench.empty:
        bench_view = bench[
            ["player", "team_short", "position", "price", "projected", "ownership"]
        ].copy()
        bench_view.columns = ["Player", "Team", "Pos", "Price £m", "Projected", "Own %"]
        st.dataframe(
            bench_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price £m": st.column_config.NumberColumn(format="£%.1f"),
                "Projected": st.column_config.NumberColumn(format="%.2f"),
                "Own %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    # ---- Full squad ----
    st.subheader("Full 15-man squad")
    full = squad[
        ["player", "team_short", "position", "price", "projected", "ownership", "form", "ep_next"]
    ].copy()
    full.columns = [
        "Player", "Team", "Pos", "Price £m", "Projected", "Own %", "Form", "EP next"
    ]
    st.dataframe(
        full,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price £m": st.column_config.NumberColumn(format="£%.1f"),
            "Projected": st.column_config.NumberColumn(format="%.2f"),
            "Own %": st.column_config.NumberColumn(format="%.1f%%"),
            "Form": st.column_config.NumberColumn(format="%.1f"),
            "EP next": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    # By club
    st.subheader("Players by club")
    by_club = (
        squad.groupby("team_short")
        .agg(Players=("player", "count"), Cost=("price", "sum"), Projected=("projected", "sum"))
        .sort_values("Players", ascending=False)
        .reset_index()
    )
    by_club.columns = ["Club", "Players", "Cost £m", "Projected"]
    st.dataframe(
        by_club,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cost £m": st.column_config.NumberColumn(format="£%.1f"),
            "Projected": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    # ---- Export ----
    st.divider()
    st.subheader("Download")

    export = squad[
        [
            "player", "first_name", "second_name", "team", "team_short",
            "position", "price", "projected", "ownership", "form", "ep_next", "ppg",
        ]
    ].copy()
    if xi is not None and not xi.empty and "player" in xi.columns:
        export["in_xi"] = export["player"].isin(xi["player"])
        if "role" in xi.columns:
            export["role"] = export["player"].map(xi.set_index("player")["role"]).fillna("")
        else:
            export["role"] = ""
    else:
        export["in_xi"] = False
        export["role"] = ""

    # Excel
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="Squad", index=False)
        xi.to_excel(writer, sheet_name="Starting XI", index=False)
        bench.to_excel(writer, sheet_name="Bench", index=False)
        by_club.to_excel(writer, sheet_name="By club", index=False)
    buffer.seek(0)

    st.download_button(
        "⬇️ Download Excel workbook",
        data=buffer.getvalue(),
        file_name="fpl_optimized_squad.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_data = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_data,
        file_name="fpl_optimized_squad.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Set your budget and optional locks in the sidebar, then click **Build optimal squad**."
    )
    st.subheader("Player pool (preview)")
    preview = players.nlargest(20, "projected")[
        ["player", "team_short", "position", "price", "projected", "ownership", "form"]
    ]
    preview.columns = ["Player", "Team", "Pos", "Price £m", "Projected", "Own %", "Form"]
    st.dataframe(preview, use_container_width=True, hide_index=True)

st.caption(
    "Decision-support tool only — not a guarantee of FPL points. "
    "Data from the public Fantasy Premier League API. "
    "Optimisation uses linear programming (PuLP) under official squad rules."
)
