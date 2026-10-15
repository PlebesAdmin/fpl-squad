"""
FPL Squad Builder engine.

Builds a legal 15-man squad under official FPL constraints using a
transparent points projection and linear optimisation (PuLP).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import time

import numpy as np
import pandas as pd
import requests

try:
    import pulp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PuLP is required. Install with: pip install pulp"
    ) from exc


FPL_BASE = "https://fantasy.premierleague.com/api"

# Official squad structure
POSITION_LIMITS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
ELEMENT_TYPE_TO_POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Starting XI minimums (FPL rules for a valid lineup)
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_SIZE = 11


@dataclass
class SquadConfig:
    budget: float = 100.0
    max_per_team: int = 3
    # Projection blend
    ep_weight: float = 0.45
    form_weight: float = 0.30
    ppg_weight: float = 0.25
    # Optional: prefer lower ownership slightly (0 = ignore)
    differential_boost: float = 0.0
    min_minutes: int = 0


class FPLClient:
    def __init__(self, cache_dir: str = "cache", cache_hours: int = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hours = cache_hours
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "FPL-Squad-Builder/1.0",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, cache_name: str):
        cache_file = self.cache_dir / cache_name
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < self.cache_hours * 3600:
                return json.loads(cache_file.read_text(encoding="utf-8"))
        try:
            r = self.session.get(f"{FPL_BASE}/{path}", timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as exc:
            if cache_file.exists():
                return json.loads(cache_file.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"Could not reach the FPL API ({path}). Try again later."
            ) from exc
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def bootstrap(self):
        return self._get("bootstrap-static/", "bootstrap-static.json")


class SquadBuilder:
    """
    Optimises a full 15-man FPL squad.

    Constraints (official):
      - 2 GKP, 5 DEF, 5 MID, 3 FWD
      - Budget limit (default £100.0m)
      - Max N players from the same club (default 3)

    Objective:
      Maximise sum of projected points (transparent blend of ep_next, form, PPG)
    """

    def __init__(self, client: Optional[FPLClient] = None, config: Optional[SquadConfig] = None):
        self.client = client or FPLClient()
        self.config = config or SquadConfig()
        self.bootstrap_data = None
        self.players: Optional[pd.DataFrame] = None

    def load(self) -> "SquadBuilder":
        self.bootstrap_data = self.client.bootstrap()
        self._build_players()
        return self

    def _build_players(self):
        b = self.bootstrap_data
        teams = pd.DataFrame(b["teams"])
        elements = pd.DataFrame(b["elements"])

        team_map = teams.set_index("id")["name"].to_dict()
        short_map = teams.set_index("id")["short_name"].to_dict()

        elements["team_name"] = elements["team"].map(team_map)
        elements["team_short"] = elements["team"].map(short_map)
        elements["position"] = elements["element_type"].map(ELEMENT_TYPE_TO_POS)
        elements["price"] = elements["now_cost"] / 10.0
        elements["ownership"] = pd.to_numeric(
            elements["selected_by_percent"], errors="coerce"
        ).fillna(0.0)

        def safe(v, default=0.0):
            try:
                if v is None or v == "":
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        cfg = self.config
        rows = []
        for _, p in elements.iterrows():
            status = str(p.get("status", "a"))
            if status in {"u", "s"}:  # unavailable / suspended
                continue
            minutes = safe(p.get("minutes"), 0)
            if minutes < cfg.min_minutes:
                continue
            price = safe(p.get("price"), 0)
            if price <= 0:
                continue

            ep = safe(p.get("ep_next"), 0)
            form = safe(p.get("form"), 0)
            ppg = safe(p.get("points_per_game"), 0)
            proj = (
                cfg.ep_weight * ep
                + cfg.form_weight * form
                + cfg.ppg_weight * ppg
            )
            # Mild differential nudge (optional)
            if cfg.differential_boost > 0:
                own = safe(p.get("ownership"), 50)
                proj *= 1.0 + cfg.differential_boost * (1.0 - min(own, 100) / 100.0)

            chance = p.get("chance_of_playing_next_round")
            if chance is None:
                avail = 1.0
            else:
                avail = max(0.0, min(1.0, safe(chance, 100) / 100.0))
            proj *= avail

            rows.append(
                {
                    "id": int(p["id"]),
                    "player": p["web_name"],
                    "first_name": p.get("first_name", ""),
                    "second_name": p.get("second_name", ""),
                    "team_id": int(p["team"]),
                    "team": p["team_name"],
                    "team_short": p["team_short"],
                    "position": p["position"],
                    "price": price,
                    "ownership": safe(p.get("ownership")),
                    "ep_next": ep,
                    "form": form,
                    "ppg": ppg,
                    "projected": max(proj, 0.01),
                    "minutes": minutes,
                    "status": status,
                    "news": p.get("news", "") or "",
                }
            )

        self.players = pd.DataFrame(rows)

    def optimise(
        self,
        locked_ids: Optional[Set[int]] = None,
        excluded_ids: Optional[Set[int]] = None,
        budget: Optional[float] = None,
        max_per_team: Optional[int] = None,
    ) -> Dict:
        """
        Solve for the optimal 15-man squad.

        Returns dict with:
          squad (DataFrame), total_cost, total_projected, bank,
          starting_xi suggestion, bench, status message
        """
        if self.players is None or self.players.empty:
            self.load()

        df = self.players.copy()
        locked_ids = locked_ids or set()
        excluded_ids = excluded_ids or set()
        budget = float(budget if budget is not None else self.config.budget)
        max_per_team = int(max_per_team if max_per_team is not None else self.config.max_per_team)

        if excluded_ids:
            df = df[~df["id"].isin(excluded_ids)].copy()

        # Validate locked players exist and fit basic rules
        locked = df[df["id"].isin(locked_ids)].copy() if locked_ids else pd.DataFrame()
        if locked_ids and len(locked) != len(locked_ids):
            missing = locked_ids - set(locked["id"].tolist())
            return {
                "ok": False,
                "error": f"Locked players not available or filtered out: {missing}",
            }

        if not locked.empty:
            # Position counts for locked
            for pos, limit in POSITION_LIMITS.items():
                n = int((locked["position"] == pos).sum())
                if n > limit:
                    return {
                        "ok": False,
                        "error": f"Too many locked {pos} players ({n} > {limit}).",
                    }
            if float(locked["price"].sum()) > budget:
                return {
                    "ok": False,
                    "error": f"Locked players alone cost £{locked['price'].sum():.1f}m (over budget).",
                }
            team_counts = locked.groupby("team_id").size()
            if (team_counts > max_per_team).any():
                return {
                    "ok": False,
                    "error": "Locked players already exceed max per club.",
                }

        # Build ILP
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
        ids = df["id"].tolist()
        x = pulp.LpVariable.dicts("p", ids, cat="Binary")

        # Objective
        proj_map = df.set_index("id")["projected"].to_dict()
        prob += pulp.lpSum(proj_map[i] * x[i] for i in ids)

        # Exactly 15 players with position limits
        for pos, limit in POSITION_LIMITS.items():
            pos_ids = df.loc[df["position"] == pos, "id"].tolist()
            prob += pulp.lpSum(x[i] for i in pos_ids) == limit, f"pos_{pos}"

        # Budget
        price_map = df.set_index("id")["price"].to_dict()
        prob += pulp.lpSum(price_map[i] * x[i] for i in ids) <= budget, "budget"

        # Max per team
        for team_id, g in df.groupby("team_id"):
            team_ids = g["id"].tolist()
            prob += (
                pulp.lpSum(x[i] for i in team_ids) <= max_per_team,
                f"team_{team_id}",
            )

        # Locked players must be selected
        for lid in locked_ids:
            if lid in x:
                prob += x[lid] == 1, f"lock_{lid}"

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
        status = prob.solve(solver)

        if pulp.LpStatus[status] != "Optimal":
            return {
                "ok": False,
                "error": f"Could not find a valid squad ({pulp.LpStatus[status]}). "
                "Try raising the budget, unlocking players, or lowering min minutes.",
            }

        chosen_ids = [i for i in ids if pulp.value(x[i]) > 0.5]
        squad = df[df["id"].isin(chosen_ids)].copy()
        squad = squad.sort_values(
            by=["position", "projected"],
            key=lambda s: s.map(POSITION_ORDER) if s.name == "position" else s,
            ascending=[True, False],
        ).reset_index(drop=True)

        total_cost = float(squad["price"].sum())
        total_proj = float(squad["projected"].sum())
        bank = budget - total_cost

        xi, bench = self._suggest_xi(squad)

        return {
            "ok": True,
            "squad": squad,
            "starting_xi": xi,
            "bench": bench,
            "total_cost": round(total_cost, 1),
            "total_projected": round(total_proj, 2),
            "bank": round(bank, 1),
            "budget": budget,
            "player_count": len(squad),
        }

    def _suggest_xi(self, squad: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pick a valid starting XI by highest projected points
        while respecting minimum position counts.
        """
        remaining = squad.sort_values("projected", ascending=False).copy()
        xi_rows = []
        counts = {p: 0 for p in POSITION_LIMITS}

        # First pass: satisfy minimums with best available
        for pos, need in XI_MIN.items():
            pool = remaining[remaining["position"] == pos]
            take = pool.head(need)
            xi_rows.append(take)
            counts[pos] += len(take)
            remaining = remaining.drop(take.index)

        # Fill to 11 with best remaining (respecting max squad counts already satisfied)
        xi = pd.concat(xi_rows) if xi_rows else pd.DataFrame()
        need = XI_SIZE - len(xi)
        if need > 0 and not remaining.empty:
            extra = remaining.head(need)
            xi = pd.concat([xi, extra])
            remaining = remaining.drop(extra.index)

        xi = xi.sort_values(
            by=["position", "projected"],
            key=lambda s: s.map(POSITION_ORDER) if s.name == "position" else s,
            ascending=[True, False],
        ).reset_index(drop=True)

        bench = remaining.sort_values(
            by=["position", "projected"],
            key=lambda s: s.map(POSITION_ORDER) if s.name == "position" else s,
            ascending=[True, False],
        ).reset_index(drop=True)

        # Captain = highest projected in XI
        if not xi.empty:
            xi = xi.copy()
            xi["role"] = ""
            cap_idx = xi["projected"].idxmax()
            xi.loc[cap_idx, "role"] = "C"
            # Vice = second highest
            others = xi.drop(index=cap_idx)
            if not others.empty:
                vc_idx = others["projected"].idxmax()
                xi.loc[vc_idx, "role"] = "VC"

        return xi, bench
