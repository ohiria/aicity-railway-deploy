"""
AICity - AI国家シミュレーション 統合版
市民エージェント + 時間管理 + 経済 + Webダッシュボード
Railway/Docker対応のシングルファイル構成
"""

import uuid
import random
import time
import threading
import json
import asyncio
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logger = logging.getLogger("aicity")

# ============================================================
# OpenRouter API統合
# ============================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/moonshotai/kimi-k2-0905")
LLM_ENABLED = bool(OPENROUTER_API_KEY)

async def llm_think(citizen_name: str, role: str, mood: str, situation: str) -> Optional[str]:
    """OpenRouter APIを使って市民の思考を生成"""
    if not LLM_ENABLED:
        return None
    try:
        prompt = f"""あなたは「{citizen_name}」という名前のAI市民です。
職業: {role}
現在の気分: {mood}
状況: {situation}

この状況で、あなたは何を考え、何をしますか？
1〜2文の短い独り言を日本語で返してください。具体的で人間味のある内容にしてください。"""

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 80,
                    "temperature": 0.9,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"LLM思考エラー ({citizen_name}): {e}")
    return None

# 非同期思考キュー（APIレート制限対策）
thought_queue: List[dict] = []
thought_results: Dict[str, str] = {}

async def process_thought_queue():
    """バックグラウンドで市民の思考を処理"""
    while True:
        if thought_queue and LLM_ENABLED:
            item = thought_queue.pop(0)
            result = await llm_think(
                item["name"], item["role"], item["mood"], item["situation"]
            )
            if result:
                thought_results[item["name"]] = result
        await asyncio.sleep(3)  # APIレート制限対策: 3秒間隔

# ============================================================
# 市民エージェント定義
# ============================================================

class AgentRole(Enum):
    FARMER = "農民"
    MERCHANT = "商人"
    CRAFTSMAN = "職人"
    TEACHER = "教師"
    POLICE = "警察官"
    CIVIL_SERVANT = "公務員"
    DOCTOR = "医者"
    ARTISAN = "工芸家"

class AgentMood(Enum):
    HAPPY = "幸福"
    NEUTRAL = "普通"
    STRESSED = "ストレス"
    ANGRY = "怒り"
    SAD = "悲しみ"

ROLE_EMOJI = {
    AgentRole.FARMER: "🌾",
    AgentRole.MERCHANT: "🏪",
    AgentRole.CRAFTSMAN: "🔨",
    AgentRole.TEACHER: "📚",
    AgentRole.POLICE: "👮",
    AgentRole.CIVIL_SERVANT: "🏛️",
    AgentRole.DOCTOR: "🏥",
    AgentRole.ARTISAN: "🎨",
}

MOOD_EMOJI = {
    AgentMood.HAPPY: "😊",
    AgentMood.NEUTRAL: "😐",
    AgentMood.STRESSED: "😰",
    AgentMood.ANGRY: "😡",
    AgentMood.SAD: "😢",
}

@dataclass
class AgentProfile:
    id: str
    name: str
    age: int
    gender: str
    role: AgentRole
    money: float
    health: int
    mood: AgentMood
    skills: Dict[str, int]
    relationships: Dict[str, float] = field(default_factory=dict)
    location: str = "中心街"

class AICitizen:
    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.memory: List[str] = []
        self.goals: List[str] = []
        self.action_queue: List[str] = []
        self.current_action: str = "待機中"
        self.personality = {
            "extroversion": random.random(),
            "agreeableness": random.random(),
            "conscientiousness": random.random(),
            "neuroticism": random.random(),
            "openness": random.random(),
        }

    def think(self, city_state: Dict) -> str:
        self._update_mood(city_state)
        self._form_goals(city_state)
        self._plan_actions()

        # LLM思考をキューに追加（低確率で、APIコスト節約）
        if LLM_ENABLED and random.random() < 0.05:
            thought_queue.append({
                "name": self.profile.name,
                "role": self.profile.role.value,
                "mood": self.profile.mood.value,
                "situation": f"時刻は{city_state.get('time', '?')}時、季節は{city_state.get('season', '?')}。{'仕事中' if city_state.get('is_work_time') else '休憩中'}。所持金¥{self.profile.money:.0f}、健康{self.profile.health}/100。",
            })

        # LLM思考結果があれば反映
        if self.profile.name in thought_results:
            self.current_action = thought_results.pop(self.profile.name)

        return f"{self.profile.name}は{self.profile.mood.value}です。"

    def _update_mood(self, city_state: Dict):
        if random.random() < 0.15:
            weights = [0.3, 0.35, 0.15, 0.1, 0.1]
            self.profile.mood = random.choices(list(AgentMood), weights=weights, k=1)[0]

    def _form_goals(self, city_state: Dict):
        self.goals = []
        if self.profile.health < 50:
            self.goals.append("健康を回復する")
        if self.profile.money < 100:
            self.goals.append("お金を稼ぐ")

        role_goals = {
            AgentRole.FARMER: "農作物を育てる",
            AgentRole.MERCHANT: "商売をする",
            AgentRole.CRAFTSMAN: "製品を作る",
            AgentRole.TEACHER: "授業を行う",
            AgentRole.POLICE: "街を巡回する",
            AgentRole.CIVIL_SERVANT: "行政業務をする",
            AgentRole.DOCTOR: "患者を診察する",
            AgentRole.ARTISAN: "作品を制作する",
        }
        self.goals.append(role_goals.get(self.profile.role, "仕事をする"))

        if random.random() < 0.3:
            self.goals.append("新しい関係を築く")

    def _plan_actions(self):
        self.action_queue = []
        work_actions = {
            AgentRole.FARMER: "畑を耕す",
            AgentRole.MERCHANT: "商品を売る",
            AgentRole.CRAFTSMAN: "木工品を作る",
            AgentRole.TEACHER: "生徒に教える",
            AgentRole.POLICE: "パトロールする",
            AgentRole.CIVIL_SERVANT: "書類を処理する",
            AgentRole.DOCTOR: "診療を行う",
            AgentRole.ARTISAN: "絵を描く",
        }
        for goal in self.goals[:3]:
            if goal == "健康を回復する":
                self.action_queue.append("医者を訪れる")
            elif goal == "お金を稼ぐ":
                self.action_queue.append(work_actions.get(self.profile.role, "仕事をする"))
            elif goal in role_goals_reverse():
                self.action_queue.append(work_actions.get(self.profile.role, "仕事をする"))
            elif goal == "新しい関係を築く":
                self.action_queue.append("他の市民と交流する")

    def execute_action(self, action: str) -> str:
        self.current_action = action
        self.memory.append(f"{action}")
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]

        # アクション効果
        money_change = random.randint(10, 80)
        health_change = random.randint(-5, 2)
        self.profile.money += money_change
        self.profile.health = max(0, min(100, self.profile.health + health_change))
        return f"{self.profile.name}は{action}。(+¥{money_change})"

    def interact(self, other: "AICitizen") -> str:
        change = random.uniform(-0.1, 0.2)
        self.profile.relationships[other.profile.id] = (
            self.profile.relationships.get(other.profile.id, 0) + change
        )
        types = ["会話した", "助け合った", "議論した", "協力した", "一緒に食事した"]
        interaction = random.choice(types)
        return f"{self.profile.name}と{other.profile.name}が{interaction}"


def role_goals_reverse():
    return {
        "農作物を育てる", "商売をする", "製品を作る", "授業を行う",
        "街を巡回する", "行政業務をする", "患者を診察する", "作品を制作する",
    }


# 市民生成
FIRST_NAMES_M = ["健一", "太郎", "翔太", "大翔", "蓮", "樹", "大和", "悠真", "陽翔", "朝陽"]
FIRST_NAMES_F = ["花子", "美咲", "愛子", "優花", "結衣", "陽菜", "凛", "咲良", "芽依", "紬"]
LAST_NAMES = ["田中", "佐藤", "鈴木", "高橋", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
              "吉田", "山田", "松本", "井上", "木村", "林", "清水", "山口", "阿部", "池田"]
LOCATIONS = ["中心街", "住宅区", "商業区", "工業区", "農業区", "学校区", "病院前", "市役所前", "公園", "市場"]


def generate_citizen() -> AICitizen:
    gender = random.choice(["男", "女"])
    first = random.choice(FIRST_NAMES_M if gender == "男" else FIRST_NAMES_F)
    name = f"{random.choice(LAST_NAMES)} {first}"
    profile = AgentProfile(
        id=str(uuid.uuid4()),
        name=name,
        age=random.randint(18, 75),
        gender=gender,
        role=random.choice(list(AgentRole)),
        money=random.uniform(200, 2000),
        health=random.randint(60, 100),
        mood=random.choice(list(AgentMood)),
        skills={r.name: random.randint(1, 100) for r in AgentRole},
        location=random.choice(LOCATIONS),
    )
    return AICitizen(profile)


# ============================================================
# 時間・経済・イベントシステム
# ============================================================

class AICityTime:
    def __init__(self, speed_multiplier: int = 60):
        self.speed_multiplier = speed_multiplier
        self.current_time = datetime(2024, 1, 1, 6, 0)
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.observers: List[Callable] = []
        self.day_counter = 0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def add_observer(self, callback: Callable):
        self.observers.append(callback)

    def _loop(self):
        while self.running:
            self.current_time += timedelta(minutes=self.speed_multiplier)
            if self.current_time.hour == 0 and self.current_time.minute == 0:
                self.day_counter += 1
                for obs in self.observers:
                    try: obs("day_change", self.current_time)
                    except: pass
            for obs in self.observers:
                try: obs("time_change", self.current_time)
                except: pass
            time.sleep(1)

    def get_time_string(self) -> str:
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        wd = weekdays[self.current_time.weekday()]
        return self.current_time.strftime(f'%Y年%m月%d日({wd}) %H:%M')

    def get_season(self) -> str:
        m = self.current_time.month
        return ["冬","冬","春","春","春","夏","夏","夏","秋","秋","秋","冬"][m-1]

    def is_work_time(self) -> bool:
        return 9 <= self.current_time.hour < 17

    def is_night(self) -> bool:
        return self.current_time.hour >= 22 or self.current_time.hour < 6


class EventBus:
    def __init__(self):
        self.events: List[dict] = []
        self.max_events = 200

    def publish(self, etype: str, data: str, citizen: str = ""):
        evt = {
            "type": etype,
            "data": data,
            "citizen": citizen,
            "timestamp": datetime.now().isoformat(),
        }
        self.events.append(evt)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def recent(self, n: int = 20) -> List[dict]:
        return list(reversed(self.events[-n:]))


class Economy:
    def __init__(self):
        self.prices = {"食料": 120, "衣料": 240, "住宅": 600, "道具": 180, "医療": 300}
        self.history: List[dict] = []

    def tick(self, season: str):
        for item in self.prices:
            change = random.uniform(-0.03, 0.04)
            if season == "冬" and item in ("食料", "衣料"):
                change += 0.02
            elif season == "夏" and item == "医療":
                change -= 0.01
            self.prices[item] = max(50, self.prices[item] * (1 + change))
        self.history.append({k: round(v) for k, v in self.prices.items()})
        if len(self.history) > 100:
            self.history = self.history[-100:]


# ============================================================
# シミュレーション本体
# ============================================================

class AICitySimulation:
    def __init__(self, pop: int = 30):
        self.citizens: List[AICitizen] = []
        self.city_time = AICityTime(speed_multiplier=60)
        self.event_bus = EventBus()
        self.economy = Economy()
        self.pop = pop
        self.tick_count = 0

    def initialize(self):
        self.citizens = [generate_citizen() for _ in range(self.pop)]
        self.city_time.add_observer(self._on_time)
        self.event_bus.publish("システム", f"{self.pop}人の市民が生成されました")

    def start(self):
        self.initialize()
        self.city_time.start()
        self.event_bus.publish("システム", "シミュレーション開始！")

    def stop(self):
        self.city_time.stop()

    def _on_time(self, etype: str, t):
        if etype == "time_change":
            self.tick_count += 1
            self._update_citizens()
        elif etype == "day_change":
            self.economy.tick(self.city_time.get_season())
            self.event_bus.publish("経済", f"日次経済更新 - 季節: {self.city_time.get_season()}")

    def _update_citizens(self):
        for c in self.citizens:
            state = {
                "time": self.city_time.current_time.hour,
                "is_work_time": self.city_time.is_work_time(),
                "is_night": self.city_time.is_night(),
                "season": self.city_time.get_season(),
            }
            c.think(state)

            if c.action_queue:
                action = c.action_queue.pop(0)
                result = c.execute_action(action)
                if random.random() < 0.3:
                    self.event_bus.publish("活動", result, c.profile.name)

            # ランダム交流
            if random.random() < 0.1 and len(self.citizens) > 1:
                other = random.choice([x for x in self.citizens if x != c])
                interaction = c.interact(other)
                if random.random() < 0.4:
                    self.event_bus.publish("交流", interaction, c.profile.name)

            # 場所移動
            if random.random() < 0.05:
                old = c.profile.location
                c.profile.location = random.choice(LOCATIONS)
                if old != c.profile.location and random.random() < 0.2:
                    self.event_bus.publish("移動", f"{c.profile.name}が{old}から{c.profile.location}へ移動", c.profile.name)

    def get_stats(self) -> dict:
        if not self.citizens:
            return {}
        return {
            "total_population": len(self.citizens),
            "average_money": sum(c.profile.money for c in self.citizens) / len(self.citizens),
            "average_health": sum(c.profile.health for c in self.citizens) / len(self.citizens),
            "active_citizens": len([c for c in self.citizens if self.city_time.is_work_time()]),
            "mood_distribution": {
                m.value: len([c for c in self.citizens if c.profile.mood == m])
                for m in AgentMood
            },
        }

    def get_citizens_data(self) -> List[dict]:
        return [
            {
                "name": c.profile.name,
                "age": c.profile.age,
                "gender": c.profile.gender,
                "role": c.profile.role.value,
                "role_emoji": ROLE_EMOJI.get(c.profile.role, ""),
                "health": c.profile.health,
                "money": round(c.profile.money),
                "mood": c.profile.mood.value,
                "mood_emoji": MOOD_EMOJI.get(c.profile.mood, ""),
                "location": c.profile.location,
                "current_action": c.current_action,
            }
            for c in self.citizens
        ]


# ============================================================
# Webダッシュボード
# ============================================================

simulation: Optional[AICitySimulation] = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulation
    simulation = AICitySimulation(pop=30)
    simulation.start()
    logger.info(f"LLM enabled: {LLM_ENABLED}")
    # バックグラウンドタスク: LLM思考処理
    task = asyncio.create_task(process_thought_queue())
    yield
    task.cancel()
    if simulation:
        simulation.stop()

app = FastAPI(title="AICity Dashboard", lifespan=lifespan)


class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg: str):
        for ws in list(self.connections):
            try:
                await ws.send_text(msg)
            except:
                self.disconnect(ws)

mgr = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/status")
async def api_status():
    return {"llm_enabled": LLM_ENABLED, "model": OPENROUTER_MODEL if LLM_ENABLED else "none", "population": len(simulation.citizens) if simulation else 0}

@app.get("/api/state")
async def api_state():
    if not simulation:
        return {}
    return {
        "current_time": simulation.city_time.get_time_string(),
        "season": simulation.city_time.get_season(),
        "day": simulation.city_time.day_counter,
        "tick": simulation.tick_count,
        "statistics": simulation.get_stats(),
        "citizens": simulation.get_citizens_data(),
        "events": simulation.event_bus.recent(30),
        "economy": {k: round(v) for k, v in simulation.economy.prices.items()},
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await mgr.connect(websocket)
    try:
        while True:
            if simulation:
                data = {
                    "current_time": simulation.city_time.get_time_string(),
                    "season": simulation.city_time.get_season(),
                    "day": simulation.city_time.day_counter,
                    "statistics": simulation.get_stats(),
                    "citizens": simulation.get_citizens_data(),
                    "events": simulation.event_bus.recent(30),
                    "economy": {k: round(v) for k, v in simulation.economy.prices.items()},
                }
                await websocket.send_text(json.dumps(data, ensure_ascii=False))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        mgr.disconnect(websocket)
    except Exception:
        mgr.disconnect(websocket)


# ============================================================
# ダッシュボードHTML
# ============================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🏙️ AICity - AI国家シミュレーション</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans JP',sans-serif;background:#0a0a1a;color:#e0e0e0;min-height:100vh}
.header{background:linear-gradient(135deg,#1a1a3e 0%,#2d1b69 50%,#1a3a5c 100%);padding:24px 32px;text-align:center;border-bottom:2px solid rgba(100,255,218,0.3);position:relative;overflow:hidden}
.header::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:radial-gradient(ellipse at 50% 0%,rgba(100,255,218,0.1) 0%,transparent 70%);pointer-events:none}
.header h1{font-size:2.2em;color:#64ffda;text-shadow:0 0 20px rgba(100,255,218,0.3);margin-bottom:8px;position:relative}
.header .sub{color:rgba(255,255,255,0.7);font-size:1em;position:relative}
.time-bar{display:flex;justify-content:center;gap:24px;padding:12px;background:rgba(0,0,0,0.3);font-size:0.95em}
.time-bar span{display:flex;align-items:center;gap:6px}
.pulse{width:8px;height:8px;border-radius:50%;background:#4caf50;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 4px #4caf50}50%{opacity:.4;box-shadow:0 0 8px #4caf50}}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:20px;max-width:1440px;margin:0 auto}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:20px;transition:transform .2s}
.card:hover{transform:translateY(-2px);border-color:rgba(100,255,218,0.3)}
.card h2{color:#64ffda;font-size:1.2em;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.full{grid-column:1/-1}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:600px){.stats{grid-template-columns:repeat(2,1fr)}}
.stat{background:rgba(0,0,0,0.3);padding:14px;border-radius:8px;text-align:center;border-left:3px solid #64ffda}
.stat .label{font-size:.8em;color:rgba(255,255,255,0.6);margin-bottom:4px}
.stat .value{font-size:1.8em;font-weight:700;color:#64ffda}
.econ-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.econ-item{background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;display:flex;justify-content:space-between;align-items:center}
.econ-item .price{font-size:1.3em;font-weight:700;color:#64ffda}
.mood-bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.mood-chip{padding:4px 10px;border-radius:12px;font-size:.85em;font-weight:600}
.mood-happy{background:rgba(76,175,80,0.3);color:#81c784}
.mood-neutral{background:rgba(255,193,7,0.3);color:#ffd54f}
.mood-stressed{background:rgba(255,152,0,0.3);color:#ffb74d}
.mood-angry{background:rgba(244,67,54,0.3);color:#e57373}
.mood-sad{background:rgba(156,39,176,0.3);color:#ba68c8}
.citizens{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;max-height:500px;overflow-y:auto;padding-right:8px}
.citizens::-webkit-scrollbar{width:6px}
.citizens::-webkit-scrollbar-thumb{background:rgba(100,255,218,0.3);border-radius:3px}
.ctz{background:rgba(0,0,0,0.25);padding:12px;border-radius:8px;border-left:3px solid rgba(100,255,218,0.5);transition:all .2s}
.ctz:hover{background:rgba(100,255,218,0.05);border-left-color:#64ffda}
.ctz .name{font-weight:700;color:#64ffda;font-size:1.05em;margin-bottom:6px}
.ctz .info{font-size:.85em;line-height:1.6;color:rgba(255,255,255,0.8)}
.ctz .action{font-size:.8em;color:#ffb74d;margin-top:4px;font-style:italic}
.ctz .loc{font-size:.75em;color:rgba(255,255,255,0.5)}
.events{max-height:350px;overflow-y:auto;padding-right:8px}
.events::-webkit-scrollbar{width:6px}
.events::-webkit-scrollbar-thumb{background:rgba(100,255,218,0.3);border-radius:3px}
.evt{padding:8px 12px;margin-bottom:6px;background:rgba(0,0,0,0.2);border-radius:6px;border-left:3px solid #64ffda;font-size:.9em;animation:fadeIn .5s ease}
.evt .type{color:#64ffda;font-weight:600;margin-right:8px}
.evt .ts{color:rgba(255,255,255,0.4);font-size:.8em;float:right}
@keyframes fadeIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
.status{position:fixed;top:12px;right:16px;padding:6px 14px;border-radius:16px;font-size:.85em;z-index:100;backdrop-filter:blur(8px)}
.status.ok{background:rgba(76,175,80,0.2);color:#81c784;border:1px solid rgba(76,175,80,0.3)}
.status.err{background:rgba(244,67,54,0.2);color:#e57373;border:1px solid rgba(244,67,54,0.3)}
.footer{text-align:center;padding:16px;color:rgba(255,255,255,0.3);font-size:.8em;border-top:1px solid rgba(255,255,255,0.05)}
</style>
</head>
<body>
<div class="status err" id="status">🔴 接続中...</div>
<div class="header">
  <h1>🏙️ AICity</h1>
  <div class="sub">AI国家シミュレーション — AIが自律的に生活する仮想都市</div>
</div>
<div class="time-bar">
  <span><span class="pulse"></span> <b id="vtime">--</b></span>
  <span>🌿 季節: <b id="season">--</b></span>
  <span>📅 経過日数: <b id="days">0</b></span>
  <span>🧠 AI: <b id="llm">確認中...</b></span>
</div>
<div class="grid">
  <div class="card">
    <h2>📊 都市統計</h2>
    <div class="stats">
      <div class="stat"><div class="label">総人口</div><div class="value" id="pop">--</div></div>
      <div class="stat"><div class="label">平均所持金</div><div class="value" id="money">--</div></div>
      <div class="stat"><div class="label">平均健康度</div><div class="value" id="health">--</div></div>
      <div class="stat"><div class="label">活動中</div><div class="value" id="active">--</div></div>
    </div>
    <div class="mood-bar" id="moods"></div>
  </div>
  <div class="card">
    <h2>💰 経済指標</h2>
    <div class="econ-grid" id="economy"></div>
  </div>
  <div class="card full">
    <h2>👥 市民一覧 <span style="font-size:.7em;color:rgba(255,255,255,0.4)">(リアルタイム更新)</span></h2>
    <div class="citizens" id="citizens"></div>
  </div>
  <div class="card full">
    <h2>📝 イベントログ</h2>
    <div class="events" id="events"></div>
  </div>
</div>
<div class="footer">AICity v1.0 — Built with FastAPI + WebSocket — AI市民が24時間365日活動中</div>
<script>
let ws, retries=0;
const $=id=>document.getElementById(id);
function connect(){
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(proto+'//'+location.host+'/ws');
  ws.onopen=()=>{
    $('status').className='status ok';
    $('status').innerHTML='🟢 接続完了';
    retries=0;
  };
  ws.onmessage=e=>{try{update(JSON.parse(e.data))}catch(err){console.error(err)}};
  ws.onclose=()=>{
    $('status').className='status err';
    $('status').innerHTML='🔴 再接続中...';
    setTimeout(connect,Math.min(3000*++retries,30000));
  };
}
const econEmoji={'食料':'🍚','衣料':'👕','住宅':'🏠','道具':'🔧','医療':'💊'};
const moodClass={'幸福':'happy','普通':'neutral','ストレス':'stressed','怒り':'angry','悲しみ':'sad'};
function update(d){
  if(d.current_time)$('vtime').textContent=d.current_time;
  if(d.season)$('season').textContent=d.season;
  if(d.day!==undefined)$('days').textContent=d.day;
  if(d.statistics){
    const s=d.statistics;
    $('pop').textContent=s.total_population||'--';
    $('money').textContent='¥'+(s.average_money||0).toFixed(0);
    $('health').textContent=(s.average_health||0).toFixed(0);
    $('active').textContent=s.active_citizens||'--';
    if(s.mood_distribution){
      $('moods').innerHTML=Object.entries(s.mood_distribution).map(([m,c])=>
        `<span class="mood-chip mood-${moodClass[m]||'neutral'}">${m}: ${c}人</span>`
      ).join('');
    }
  }
  if(d.economy){
    $('economy').innerHTML=Object.entries(d.economy).map(([k,v])=>
      `<div class="econ-item"><span>${econEmoji[k]||'📦'} ${k}</span><span class="price">¥${v}</span></div>`
    ).join('');
  }
  if(d.citizens){
    $('citizens').innerHTML=d.citizens.map(c=>`
      <div class="ctz">
        <div class="name">${c.role_emoji} ${c.name} <span style="font-size:.75em;color:rgba(255,255,255,0.4)">${c.age}歳・${c.gender}</span></div>
        <div class="info">
          職業: ${c.role} | 健康: ${c.health}/100 | 所持金: ¥${c.money}<br>
          気分: ${c.mood_emoji} ${c.mood}
        </div>
        <div class="action">💭 ${c.current_action}</div>
        <div class="loc">📍 ${c.location}</div>
      </div>
    `).join('');
  }
  if(d.events){
    $('events').innerHTML=d.events.map(e=>
      `<div class="evt"><span class="type">[${e.type}]</span>${e.data}</div>`
    ).join('');
  }
}
connect();
fetch('/api/status').then(r=>r.json()).then(d=>{
  $('llm').textContent=d.llm_enabled?'✅ '+d.model:'⚠️ ルールベース';
  $('llm').style.color=d.llm_enabled?'#81c784':'#ffb74d';
}).catch(()=>{$('llm').textContent='⚠️ 未接続'});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
