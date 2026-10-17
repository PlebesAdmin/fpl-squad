"""
FPL Squad Builder engine.

Builds a legal 15-man squad under official FPL constraints.

Uses a transparent greedy optimiser by default (works on all Streamlit
environments). Optionally tries PuLP if available — never uses
LpVariable.dicts (broken on some Python 3.14 / Streamlit images).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import time

import pandas as pd
import requests

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    pulp = None  # type: ignore


FPL_BASE = "https://fantasy.premierleague.com/api"

POSITION_LIMITS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
ELEMENT_TYPE_TO_POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_SIZE = 11


@dataclass
class SquadConfig:
    budget: float = 100.0
    max_per_team: int = 3
    ep_weight: float = 0.45
    form_weight: float = 0.30
    ppg_weight: float = 0.25
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
                "User-Agent": "FPL-Squad-Builder/1.2",
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
    def __init__(
        self,
        client: Optional[FPLClient] = None,
        config: Optional[SquadConfig] = None,
    ):
        self.client = client or FPLClient()
        self.config = config or SquadConfig()
        self.bootstrap_data = None
        self.players: Optional[pd.DataFrame] = None

    def load(self) -> "SquadBuilder":
        self.bootstrap_data = self.client.bootstrap()
        self._build_players()
        return self

    def _safe(self, v, default=0.0):
        try:
            if v is None or v == "":
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

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

        cfg = self.config
        rows = []
        for _, p in elements.iterrows():
            status = str(p.get("status", "a"))
            if status in {"u", "s"}:
                continue
            minutes = self._safe(p.get("minutes"), 0)
            if minutes < cfg.min_minutes:
                continue
            price = self._safe(p.get("price"), 0)
            if price <= 0:
                continue

            ep = self._safe(p.get("ep_next"), 0)
            form = self._safe(p.get("form"), 0)
            ppg = self._safe(p.get("points_per_game"), 0)
            proj = (
                cfg.ep_weight * ep
                + cfg.form_weight * form
                + cfg.ppg_weight * ppg
            )
            if cfg.differential_boost > 0:
                own = self._safe(p.get("ownership"), 50)
                proj *= 1.0 + cfg.differential_boost * (
                    1.0 - min(own, 100) / 100.0
                )

            chance = p.get("chance_of_playing_next_round")
            if chance is None:
                avail = 1.0
            else:
                avail = max(0.0, min(1.0, self._safe(chance, 100) / 100.0))
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
                    "ownership": self._safe(p.get("ownership")),
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
        if self.players is None or self.players.empty:
            self.load()

        df = self.players.copy()
        locked_ids = set(locked_ids or [])
        excluded_ids = set(excluded_ids or [])
        budget = float(budget if budget is not None else self.config.budget)
        max_per_team = int(
            max_per_team if max_per_team is not None else self.config.max_per_team
        )

        if excluded_ids:
            df = df[~df["id"].isin(excluded_ids)].copy()

        locked = df[df["id"].isin(locked_ids)].copy() if locked_ids else pd.DataFrame()
        if locked_ids and len(locked) != len(locked_ids):
            missing = locked_ids - set(locked["id"].tolist())
            return {
                "ok": False,
                "error": f"Locked players not available or filtered out: {sorted(missing)}",
            }

        if not locked.empty:
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

        for pos, limit in POSITION_LIMITS.items():
            n = int((df["position"] == pos).sum())
            if n < limit:
                return {
                    "ok": False,
                    "error": f"Not enough {pos} players after filters ({n} < {limit}). "
                    "Lower min minutes or remove excludes.",
                }

        # Primary: greedy (always works on Streamlit Cloud)
        result = self._optimise_greedy(df, locked_ids, budget, max_per_team)
        if result.get("ok"):
            return result

        if HAS_PULP:
            try:
                pulp_result = self._optimise_pulp(df, locked_ids, budget, max_per_team)
                if pulp_result.get("ok"):
                    return pulp_result
            except Exception:
                pass

        return result

    def _optimise_pulp(
        self,
        df: pd.DataFrame,
        locked_ids: Set[int],
        budget: float,
        max_per_team: int,
    ) -> Dict:
        """PuLP path without LpVariable.dicts (Streamlit / Python 3.14 safe)."""
        ids = [int(i) for i in df["id"].tolist()]
        if not ids:
            return {"ok": False, "error": "No players available."}

        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

        # Create binary variables one-by-one (no .dicts)
        x = {}
        for i in ids:
            x[i] = pulp.LpVariable(f"p_{i}", cat="Binary")

        proj_map = df.set_index("id")["projected"].to_dict()
        price_map = df.set_index("id")["price"].to_dict()

        prob += pulp.lpSum(
            (proj_map[i] + 0.001 * price_map[i]) * x[i] for i in ids
        )

        for pos, limit in POSITION_LIMITS.items():
            pos_ids = [int(i) for i in df.loc[df["position"] == pos, "id"].tolist()]
            if not pos_ids:
                return {"ok": False, "error": f"No {pos} players left after filters."}
            prob += pulp.lpSum(x[i] for i in pos_ids) == limit

        prob += pulp.lpSum(price_map[i] * x[i] for i in ids) <= budget + 1e-6

        for team_id, g in df.groupby("team_id"):
            team_ids = [int(i) for i in g["id"].tolist()]
            prob += pulp.lpSum(x[i] for i in team_ids) <= max_per_team

        for lid in locked_ids:
            if lid in x:
                prob += x[lid] == 1

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=45)
        status = prob.solve(solver)

        if pulp.LpStatus[status] != "Optimal":
            return {"ok": False, "error": f"Solver status: {pulp.LpStatus[status]}"}

        chosen_ids = [i for i in ids if (pulp.value(x[i]) or 0) > 0.5]
        return self._pack_result(df, chosen_ids, budget)

    def _optimise_greedy(
        self,
        df: pd.DataFrame,
        locked_ids: Set[int],
        budget: float,
        max_per_team: int,
    ) -> Dict:
        selected: List[int] = []
        cost = 0.0
        pos_count = {p: 0 for p in POSITION_LIMITS}
        team_count: Dict[int, int] = {}

        by_id = df.set_index("id")

        def can_add(pid: int) -> bool:
            if pid in selected or pid not in by_id.index:
                return False
            row = by_id.loc[pid]
            pos = row["position"]
            if pos_count[pos] >= POSITION_LIMITS[pos]:
                return False
            tid = int(row["team_id"])
            if team_count.get(tid, 0) >= max_per_team:
                return False
            if cost + float(row["price"]) > budget + 1e-9:
                return False
            return True

        def add(pid: int) -> None:
            nonlocal cost
            row = by_id.loc[pid]
            selected.append(int(pid))
            cost += float(row["price"])
            pos_count[row["position"]] += 1
            tid = int(row["team_id"])
            team_count[tid] = team_count.get(tid, 0) + 1

        for lid in locked_ids:
            if lid in by_id.index:
                if not can_add(lid):
                    return {
                        "ok": False,
                        "error": "Cannot place all locked players under current constraints.",
                    }
                add(lid)

        candidates = df[~df["id"].isin(selected)].copy()
        candidates["value"] = candidates["projected"] / candidates["price"].clip(
            lower=0.1
        )
        candidates = candidates.sort_values(
            ["projected", "value"], ascending=[False, False]
        )

        for pos, limit in POSITION_LIMITS.items():
            while pos_count[pos] < limit:
                pool = candidates[
                    (candidates["position"] == pos)
                    & (~candidates["id"].isin(selected))
                ]
                placed = False
                for _, row in pool.iterrows():
                    if can_add(int(row["id"])):
                        add(int(row["id"]))
                        placed = True
                        break
                if not placed:
                    return {
                        "ok": False,
                        "error": f"Could not fill {pos} under budget/club limits. "
                        "Try raising the budget or unlocking players.",
                    }

        selected_set = set(selected)
        for _ in range(2):
            improved = False
            for out_id in list(selected):
                out = by_id.loc[out_id]
                selected.remove(out_id)
                selected_set.discard(out_id)
                cost -= float(out["price"])
                pos_count[out["position"]] -= 1
                tid = int(out["team_id"])
                team_count[tid] = team_count.get(tid, 0) - 1
                if team_count.get(tid, 0) <= 0:
                    team_count.pop(tid, None)

                best_in = None
                best_proj = float(out["projected"])
                pool = candidates[
                    (candidates["position"] == out["position"])
                    & (~candidates["id"].isin(selected_set))
                ]
                for _, row in pool.iterrows():
                    pid = int(row["id"])
                    if float(row["projected"]) <= best_proj + 1e-9:
                        continue
                    if can_add(pid):
                        best_in = pid
                        best_proj = float(row["projected"])
                        break

                if best_in is not None:
                    add(best_in)
                    selected_set.add(best_in)
                    improved = True
                else:
                    add(out_id)
                    selected_set.add(out_id)

            if not improved:
                break

        if len(selected) != 15:
            return {
                "ok": False,
                "error": f"Builder ended with {len(selected)} players (need 15).",
            }

        return self._pack_result(df, selected, budget)

    def _pack_result(
        self, df: pd.DataFrame, chosen_ids: List[int], budget: float
    ) -> Dict:
        squad = df[df["id"].isin(chosen_ids)].copy()
        squad["_pos_ord"] = squad["position"].map(POSITION_ORDER)
        squad = (
            squad.sort_values(["_pos_ord", "projected"], ascending=[True, False])
            .drop(columns=["_pos_ord"])
            .reset_index(drop=True)
        )

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
        remaining = squad.sort_values("projected", ascending=False).copy()
        xi_parts = []

        for pos, need in XI_MIN.items():
            pool = remaining[remaining["position"] == pos]
            take = pool.head(need)
            xi_parts.append(take)
            remaining = remaining.drop(index=take.index, errors="ignore")

        xi = pd.concat(xi_parts) if xi_parts else pd.DataFrame()
        need = XI_SIZE - len(xi)
        if need > 0 and not remaining.empty:
            extra = remaining.head(need)
            xi = pd.concat([xi, extra])
            remaining = remaining.drop(index=extra.index, errors="ignore")

        if not xi.empty:
            xi = xi.copy()
            xi["_pos_ord"] = xi["position"].map(POSITION_ORDER)
            xi = (
                xi.sort_values(["_pos_ord", "projected"], ascending=[True, False])
                .drop(columns=["_pos_ord"])
                .reset_index(drop=True)
            )
            xi["role"] = ""
            cap_idx = xi["projected"].idxmax()
            xi.loc[cap_idx, "role"] = "C"
            others = xi.drop(index=cap_idx)
            if not others.empty:
                vc_idx = others["projected"].idxmax()
                xi.loc[vc_idx, "role"] = "VC"

        bench = remaining.copy()
        if not bench.empty:
            bench["_pos_ord"] = bench["position"].map(POSITION_ORDER)
            bench = (
                bench.sort_values(["_pos_ord", "projected"], ascending=[True, False])
                .drop(columns=["_pos_ord"])
                .reset_index(drop=True)
            )

        return xi, bench
