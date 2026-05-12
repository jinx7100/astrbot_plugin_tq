import os, re, json, datetime, time, requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.interpolate import make_interp_spline
from PIL import Image as ImageW

from astrbot.api.event import AstrMessageEvent, filter, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.event.filter import command
from astrbot.api.all import *
from astrbot.core.platform.message_session import MessageSession

GEO_API  = "https://geocoding-api.open-meteo.com/v1/search"
WEA_API  = "https://api.open-meteo.com/v1/forecast"

# ── WMO 天气码 ──
# (中文名, 图标分类, 带伞标志, 极端标志)
WMO = {
    0:( "晴","sunny",0,0), 1:( "晴间多云","partly_cloudy",0,0),
    2:( "局部多云","partly_cloudy",0,0), 3:( "多云","cloudy",0,0),
    45:( "雾","foggy",0,0), 48:( "大雾","foggy",0,0),
    51:( "小毛毛雨","rainy",1,0), 53:( "中毛毛雨","rainy",1,0),
    55:( "大毛毛雨","rainy",1,0), 56:( "冻毛毛雨","rainy",1,0),
    57:( "冻毛毛雨","rainy",1,0), 61:( "小雨","rainy",1,0),
    63:( "中雨","rainy",1,0), 65:( "大雨","rainy",1,1),
    66:( "冻雨","rainy",1,1), 67:( "冻雨","rainy",1,1),
    71:( "小雪","snowy",1,0), 73:( "中雪","snowy",1,0),
    75:( "大雪","snowy",1,1), 77:( "雪粒","snowy",1,0),
    80:( "阵雨","rainy",1,0), 81:( "中阵雨","rainy",1,0),
    82:( "大阵雨","rainy",1,1), 85:( "小阵雪","snowy",1,0),
    86:( "大阵雪","snowy",1,1), 95:( "雷暴","rainy",1,1),
    96:( "雷暴加冰雹","rainy",1,1), 99:( "雷暴加大冰雹","rainy",1,1),
}

ICONS = {
    "sunny":"sunny.png","partly_cloudy":"partly_cloudy.png",
    "cloudy":"cloudy.png","windy":"windy.png",
    "foggy":"foggy.png","rainy":"rainy.png","snowy":"snowy.png",
}

# ── 缓存 ──
_cache = {}
CACHE_TTL = 1800  # 30 分钟

def _cache_get(key):
    global _cache
    ent = _cache.get(key)
    if ent and time.time() - ent["ts"] < CACHE_TTL:
        return ent["data"]
    return None

def _cache_set(key, data):
    global _cache
    _cache[key] = {"ts": time.time(), "data": data}


# ── 工具函数 ──

def _geo_search(name):
    """调用 Open-Meteo 地理编码，优先返回中国结果"""
    try:
        r = requests.get(GEO_API, params={"name":name,"count":5,"language":"zh","format":"json"}, timeout=10)
        d = r.json().get("results")
        if not d: return None
        for item in d:
            if item.get("country_code") == "CN":
                return item
        return d[0]
    except: pass
    return None

def _geo(q):
    """
    地理编码 - 逐级 fallback：
      1. 原始查询
      2. 去掉末尾 市/区/县
      3. 纯中文无分隔符 => 按 2 字切分（如"广州番禺"->["广州","番禺"]）
      4. 按 市/区/县/空格 正则拆分
      5. 末尾加"市"（修正"北京"搜成重庆的问题）
    """
    q = q.strip()
    if not q: return None

    # 1. 原始查询
    ret = _geo_search(q)
    if ret: return ret

    # 2. 去掉末尾 市/区/县
    cleaned = re.sub(r'[市区县]$', '', q)
    if cleaned and cleaned != q:
        ret = _geo_search(cleaned)
        if ret: return ret

    # 3. 纯中文无分隔符 => 按 2 字切分
    if re.fullmatch(r'[\u4e00-\u9fff]+', q) and len(q) >= 4:
        for split_pos in range(2, len(q) - 1):
            left, right = q[:split_pos], q[split_pos:]
            candidates = [p for p in (left, right) if len(p) >= 2]
            for p in candidates:
                ret = _geo_search(p)
                if ret: return ret

    # 4. 按 市/区/县/空格 正则拆分
    parts = re.split(r'[市区县\s]+', q)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        for p in parts:
            ret = _geo_search(p)
            if ret: return ret

    # 5. 末尾加"市"
    if not q.endswith("市"):
        ret = _geo_search(q + "市")
        if ret: return ret

    return None

def _geo_fmt(d):
    n, a1, a2, c = d.get("name",""), d.get("admin1",""), d.get("admin2",""), d.get("country","")
    if "中国" in c:
        if a2 and a2 != n: return f"{a1}{a2}市{n}区" if a1 else n
        return f"{a1}{n}市" if a1 and a1 != n else n
    parts = [p for p in [c,a1,a2,n] if p and p!=n]
    if n not in parts: parts.append(n)
    return " ".join(parts)

def _fetch_all(lat, lon):
    """获取当前实时 + 3天逐日预报 + 24小时逐时"""
    now = None
    daily = None
    hourly = None
    try:
        r = requests.get(WEA_API, params={
            "latitude":lat,"longitude":lon,
            "current":"temperature_2m,relative_humidity_2m,apparent_temperature,weathercode,precipitation,wind_speed_10m",
            "daily":"temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,precipitation_probability_max",
            "timezone":"auto","forecast_days":3,
        }, timeout=10).json()
        now = r.get("current")
        daily = r.get("daily")
    except: pass

    try:
        r = requests.get(WEA_API, params={
            "latitude":lat,"longitude":lon,
            "hourly":"temperature_2m,weathercode,precipitation_probability",
            "timezone":"auto","forecast_hours":24,
        }, timeout=10).json()
        hourly = r.get("hourly")
    except: pass

    return now, daily, hourly


def _build_report(loc_name, current, daily, hourly):
    lines = []
    now_t = time.time()

    # ── 标题 ──
    lines.append(f"🌤 【{loc_name} 天气预报】")
    lines.append("")

    # ── 当前实况 ──
    if current:
        cd = current.get("weathercode", -1)
        w = WMO.get(cd, ("未知","not_supported",0,0))
        tmp = current.get("temperature_2m")
        feel = current.get("apparent_temperature")
        hum = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        prec = current.get("precipitation", 0)

        lines.append(f"📍 当前实况")
        lines.append(f"  天气：{w[0]}")
        if tmp is not None:
            lines.append(f"  温度：{tmp:.1f}°C（体感 {feel:.1f}°C）" if feel else f"  温度：{tmp:.1f}°C")
        if hum is not None: lines.append(f"  湿度：{hum}%")
        if wind is not None: lines.append(f"  风速：{wind:.1f} km/h")
        if prec and prec > 0: lines.append(f"  降水：{prec} mm")
        lines.append("")

        # 极端预警（当前）
        if w[3]:
            lines.append(f"⚠️ 极端天气预警：当前正在经历【{w[0]}】，请注意安全！")
            lines.append("")

    # ── 3天预报 ──
    if daily:
        lines.append(f"📅 未来3天预报")
        for i, d_ts in enumerate(daily.get("time", [])):
            wc = daily.get("weathercode", [])[i] if daily.get("weathercode") else -1
            w = WMO.get(wc, ("未知","not_supported",0,0))
            tmax = daily.get("temperature_2m_max", [])[i] if daily.get("temperature_2m_max") else None
            tmin = daily.get("temperature_2m_min", [])[i] if daily.get("temperature_2m_min") else None
            psum = daily.get("precipitation_sum", [])[i] if daily.get("precipitation_sum") else 0
            pprob = daily.get("precipitation_probability_max", [])[i] if daily.get("precipitation_probability_max") else 0

            label = "今日" if i == 0 else ("明日" if i == 1 else f"{d_ts[-5:]}")

            line = f"  {label}: {w[0]}"
            if tmax is not None: line += f" {tmin:.0f}~{tmax:.0f}°C"
            if pprob and pprob > 0: line += f" 🌧{pprob}%"
            if psum and psum > 0: line += f" (降水{psum}mm)"
            lines.append(line)

            if w[3]:
                lines.append(f"  ⚠️ 极端天气预警：{label}预计有【{w[0]}】！")
            if pprob and pprob >= 50:
                lines.append(f"  🌂 降雨概率{pprob}%，建议带伞！")
        lines.append("")

    # ── 24小时逐时（带伞判断用） ──
    if hourly:
        ts_list = hourly.get("time", [])
        cd_list = hourly.get("weathercode", [])
        pr_list = hourly.get("precipitation_probability", [0]*len(ts_list))

        need_umb = False
        ext_alerts = []
        max_precip_prob = 0
        for t_raw, cd_raw, pr_raw in zip(ts_list, cd_list, pr_list):
            h_dt = datetime.datetime.fromisoformat(t_raw)
            if (h_dt.timestamp() - now_t) > 43200: break
            w = WMO.get(cd_raw, ("未知","not_supported",0,0))
            if w[2]: need_umb = True
            if w[3]:
                ext_alerts.append(f"  ⏰ {h_dt.strftime('%H:%M')} 预计有【{w[0]}】")
            if pr_raw and pr_raw > max_precip_prob: max_precip_prob = pr_raw

        if need_umb or max_precip_prob >= 50:
            lines.append(f"🌂 未来12小时降雨概率最高 {max_precip_prob}%")
            if need_umb or max_precip_prob >= 50:
                lines.append(f"  └ 建议出门带伞！")
            lines.append("")

        if ext_alerts:
            lines.append("⚠️ 逐时极端预警：")
            lines.extend(ext_alerts)
            lines.append("")

    # ── 底部提示 ──
    lines.append("---")
    lines.append(f"🕐 更新于 {datetime.datetime.now().strftime('%H:%M')} | 每30分钟自动更新")

    return "\n".join(lines)


def _gen_chart(loc_name, hourly_dict, sid, pdir):
    """生成24小时温度折线图"""
    try:
        times = [datetime.datetime.fromisoformat(h["t"]).strftime("%H:%M") for h in hourly_dict]
        temps = [h["tmp"] for h in hourly_dict]
        icons = [h["icon"] for h in hourly_dict]
        x = np.arange(len(times))
        y = np.array(temps)
        xs = np.linspace(x.min(), x.max(), 300)
        sp = make_interp_spline(x, y, k=3 if len(temps)>=4 else 1)
        ys = sp(xs)
        # 跨平台中文字体探测
        _font_candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        fp = None
        for _f in _font_candidates:
            if os.path.exists(_f):
                fp = _f
                break
        if not fp:
            fp = os.path.join(pdir, "SourceHanSansCN-Regular.otf")
        pr = fm.FontProperties(fname=fp, size=12)
        fig, ax = plt.subplots(figsize=(16,9), facecolor="#F5F5F5")
        ax.plot(xs, ys, color="#E74C3C", lw=2, zorder=10)
        ax.set_xticks(x)
        ax.set_xticklabels(times, fontproperties=pr)
        ax.set_xlabel("时间", fontproperties=pr, fontsize=14)
        ax.set_ylabel("温度 (°C)", fontproperties=pr, fontsize=14)
        ax.set_title(f"{loc_name} 未来24小时温度变化", fontproperties=pr, fontsize=20, pad=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        def _ic(cat):
            p = os.path.join(pdir, "icons", ICONS.get(cat,"not_supported.png"))
            if not os.path.exists(p):
                p = os.path.join(pdir, "icons", "not_supported.png")
            return OffsetImage(plt.imread(p), zoom=0.2)
        tr = max(temps)-min(temps) if max(temps)!=min(temps) else 1
        md = np.median(temps)
        for xi,yi,ic in zip(x,temps,icons):
            ax.add_artist(AnnotationBbox(_ic(ic), (xi,yi),
                xybox=(0, tr*0.15*(1 if yi>md else -1)),
                boxcoords="offset points", box_alignment=(0.5,0.5),
                frameon=False, zorder=30))
            ax.text(xi, yi+0.3, f"{yi:.1f}°C", ha="center", va="bottom",
                fontproperties=pr, color="#2C3E50", fontsize=16, zorder=30)
        plt.subplots_adjust(bottom=0.15)
        safe = sid.replace(":","").replace("/","_").replace("\\","_")
        png = os.path.join(pdir, f"{safe}_tq.png")
        jpg = os.path.join(pdir, f"{safe}_tq.jpg")
        plt.savefig(png, dpi=300, bbox_inches="tight"); plt.close()
        ImageW.open(png).convert("RGB").save(jpg, quality=95)
        os.remove(png)
        return jpg
    except Exception as e:
        logger.error(f"出图失败: {e}")
        return None


def _extract_arg(event: AstrMessageEvent) -> str:
    """从消息中取出去掉指令前缀后的参数"""
    try:
        msg = str(event.get_message_str()).strip()
        parts = msg.split(maxsplit=1)
        if len(parts) < 2:
            return ""
        return parts[1].strip()
    except:
        return ""


def _extract_location_nlp(text: str) -> str | None:
    """从自然语言文本中提取天气查询地点关键词

    匹配模式（按优先级）：
      1. "X天气"、"X下雨"等 —— 地点后紧跟天气词
      2. "在X"、"到X"、"去X"等 —— 介词+地点
      3. "X那边"、"X这里"等 —— 地点+方位词
    """
    if not text:
        return None
    # 模式1: "广州天气"、"番禺下雨"、"北京温度"、"上海气温"
    m = re.search(r'([\u4e00-\u9fff]{2,5})(?:的)?(?:天气|下雨|下雪|温度|气温|台风|大风|暴雨|雷暴|雾|雪|晴|阴)', text)
    if m:
        return m.group(1)
    # 模式2: "在广州"、"去北京"、"回番禺"、"到上海"
    m = re.search(r'(?:在|到|去|回|上|下)([\u4e00-\u9fff]{2,5})(?:的)?(?:天气|下雨|温度|那|这|如何|怎样|怎么样)?', text)
    if m:
        return m.group(1)
    # 模式3: "广州那边"、"北京这里"
    m = re.search(r'([\u4e00-\u9fff]{2,5})(?:那边|这里|那里|这边)', text)
    if m:
        return m.group(1)
    return None


# ══════════════════════════════════════
#  插件主类
# ══════════════════════════════════════

@register("理予的天气", "理予", "查天气、3天预报、极端预警、自动带伞提醒、自然语言天气识别", "2.2.0", "")
class TqPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config or {}
        self.default_loc = self.config.get("default_location", "番禺")
        self.start_time = time.time()
        self._cache_hit = 0
        self._cache_miss = 0
        self._query_count = 0
        self._nlp_query_count = 0
        self._reminder_job_id = None

    # ── 配置读取辅助 ──

    def _get_cfg(self, key: str, default=None):
        """读取配置，同时兼容扁平 key 和 _conf_schema 嵌套格式。"""
        # 1. 直接取扁平 key
        val = self.config.get(key)
        if val is not None:
            return val
        # 2. 遍历所有已知配置段
        for section_key in ["reminder_settings", "nlp_weather", "weather"]:
            section = self.config.get(section_key)
            if isinstance(section, dict):
                val = section.get(key)
                if val is not None:
                    return val
                # 3. 也查 section.key 格式
                dot_prefix = f"{section_key}."
                if key.startswith(dot_prefix):
                    sub = key[len(dot_prefix):]
                    val = section.get(sub)
                    if val is not None:
                        return val
        return default

    # ── 定时提醒 ──

    async def _register_reminder(self):
        """根据配置注册天气提醒定时任务（防重复注册）"""
        enabled = self._get_cfg("reminder_enabled", False)
        if not enabled:
            logger.info("天气提醒未启用，跳过定时注册。")
            return
        cron = self._get_cfg("reminder_cron", "0 7 * * *")
        cm = getattr(self.context, "cron_manager", None)
        if not cm:
            logger.warning("cron_manager 不可用，天气提醒注册失败。")
            return
        # 🔍 防重复：检查是否已有同名任务存在（热重载时旧任务还在调度器里）
        existing_jobs = await cm.list_jobs(job_type="basic")
        for j in existing_jobs:
            if j.name == "天气提醒":
                logger.info(f"天气提醒任务已存在（job_id={j.job_id}），跳过注册。")
                self._reminder_job_id = j.job_id
                # 如果 cron 表达式变了，顺手更新
                if j.cron_expression != cron:
                    logger.info(f"检测到 cron 表达式变化：{j.cron_expression} → {cron}，更新任务。")
                    await cm.update_job(j.job_id, cron_expression=cron)
                return
        try:
            job = await cm.add_basic_job(
                name="天气提醒",
                cron_expression=cron,
                handler=self._send_weather_reminder,
                description="每日天气定时推送",
                timezone="Asia/Shanghai",
                persistent=True,
            )
            self._reminder_job_id = job.job_id
            logger.info(f"✅ 天气提醒定时任务已注册: {cron}")
        except Exception as e:
            logger.error(f"❌ 注册天气提醒定时任务失败: {e}")

    async def _unregister_reminder(self):
        """注销天气提醒定时任务"""
        if self._reminder_job_id:
            cm = getattr(self.context, "cron_manager", None)
            if cm:
                try:
                    await cm.delete_job(self._reminder_job_id)
                    logger.info("✅ 天气提醒定时任务已注销。")
                except Exception as e:
                    logger.error(f"注销天气提醒定时任务失败: {e}")
            self._reminder_job_id = None

    async def _send_weather_reminder(self):
        """定时任务执行体：查天气 → 生成报告 → 推送到所有配置目标"""
        pdir = os.path.dirname(os.path.abspath(__file__))
        location = self._get_cfg("reminder_location", self.default_loc)
        targets = self._get_cfg("reminder_targets", [])
        if not targets:
            logger.warning("天气提醒：未配置推送目标，跳过。")
            return

        result = self._query_location(location)
        if not result:
            logger.error(f"天气提醒：查询不到地点「{location}」")
            return

        loc_name = result["loc_name"]
        current = result["current"]
        daily = result["daily"]
        hourly_raw = result["hourly_raw"]

        report = _build_report(loc_name, current, daily, hourly_raw)

        # 生成图表
        hourly_dict = []
        if hourly_raw:
            for t, tmp, cd, pr in zip(
                hourly_raw["time"], hourly_raw["temperature_2m"],
                hourly_raw["weathercode"],
                hourly_raw.get("precipitation_probability", [0] * len(hourly_raw["time"]))
            ):
                w = WMO.get(cd, ("未知", "not_supported", 0, 0))
                hourly_dict.append({"t": t, "tmp": tmp, "icon": w[1]})
        img = None
        if hourly_dict:
            img = _gen_chart(loc_name, hourly_dict, f"reminder_{int(time.time())}", pdir)

        # 遍历推送
        for target in targets:
            try:
                chain = MessageChain([Plain(report)])
                if img:
                    chain.chain(Image.fromFileSystem(img))
                await self.context.send_message(target, chain)
                logger.info(f"天气提醒已推送到 {target}")
            except Exception as e:
                logger.error(f"天气提醒推送 {target} 失败: {e}")

        if img:
            try:
                os.remove(img)
            except Exception:
                pass

    # ── 生命周期 ──

    async def initialize(self):
        """插件激活时调用"""
        await self._register_reminder()

    async def terminate(self):
        """插件禁用/重载时调用"""
        await self._unregister_reminder()

    # ── 核心查询 ──

    def _query_location(self, q: str):
        """查询某地天气，走缓存"""
        key = q.strip()
        cached = _cache_get(key)
        if cached:
            self._cache_hit += 1
            return cached
        self._cache_miss += 1
        g = _geo(q)
        if not g: return None
        loc_name = _geo_fmt(g)
        now, daily, hourly_raw = _fetch_all(g["latitude"], g["longitude"])
        if not now and not daily: return None
        result = {
            "loc_name": loc_name,
            "lat": g["latitude"], "lon": g["longitude"],
            "current": now, "daily": daily,
            "hourly_raw": hourly_raw,
        }
        _cache_set(key, result)
        _cache_set(loc_name, result)
        return result

    # ── 命令: /天气 ──

    @command("天气")
    async def cmd_weather(self, event: AstrMessageEvent):
        self._query_count += 1
        pdir = os.path.dirname(os.path.abspath(__file__))

        args = _extract_arg(event)
        if args and args not in ("怎么样","咋样","怎样","如何","?"):
            q = args
        else:
            q = self.default_loc

        result = self._query_location(q)
        if not result:
            yield event.chain_result([Plain(f"没找到「{q}」，换个说法试试？")])
            return

        loc_name = result["loc_name"]
        current  = result["current"]
        daily    = result["daily"]
        hourly_raw = result["hourly_raw"]

        report = _build_report(loc_name, current, daily, hourly_raw)

        hourly_dict = []
        if hourly_raw:
            for t,tmp,cd,pr in zip(hourly_raw["time"], hourly_raw["temperature_2m"],
                                     hourly_raw["weathercode"],
                                     hourly_raw.get("precipitation_probability",[0]*len(hourly_raw["time"]))):
                w = WMO.get(cd, ("未知","not_supported",0,0))
                hourly_dict.append({"t":t,"tmp":tmp,"icon":w[1]})
        img = None
        if hourly_dict:
            img = _gen_chart(loc_name, hourly_dict, event.unified_msg_origin, pdir)

        if img:
            yield event.chain_result([Plain(report + "\n"), Image.fromFileSystem(img)])
            try: os.remove(img)
            except: pass
        else:
            yield event.chain_result([Plain(report)])

    # ── 命令: /天气状态 ──

    @command("天气状态")
    async def cmd_status(self, event: AstrMessageEvent):
        upt = time.time() - self.start_time
        h, r = divmod(int(upt), 3600)
        m, s = divmod(r, 60)
        upt_str = f"{h}时{m}分"
        total = self._cache_hit + self._cache_miss
        hit_rate = f"{self._cache_hit/total*100:.1f}%" if total else "N/A"
        cache_keys = list(_cache.keys())
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 提醒状态
        reminder_enabled = self._get_cfg("reminder_enabled", False)
        reminder_cron = self._get_cfg("reminder_cron", "0 7 * * *")
        reminder_loc = self._get_cfg("reminder_location", self.default_loc)
        reminder_targets = self._get_cfg("reminder_targets", [])
        reminder_status = "✅ 已启用" if (reminder_enabled and self._reminder_job_id) else "⛔ 未启用"

        # 自然语言识别状态
        nlp_enabled = self._get_cfg("nlp_weather_enabled", False)
        nlp_fallback = self._get_cfg("nlp_weather_fallback", "广州")
        nlp_status = "✅ 已启用" if nlp_enabled else "⛔ 未启用"

        lines = [
            f"📊 【理予的天气 - 运行状态】",
            f"",
            f"🕐 当前时间：{now_str}",
            f"⏱ 运行时长：{upt_str}",
            f"📥 查询次数：{self._query_count}",
            f"💬 NLP触发：{self._nlp_query_count} 次",
            f"💾 缓存命中：{self._cache_hit} / {total}（命中率 {hit_rate}）",
            f"🗂 缓存条目：{len(cache_keys)} 条",
            f"📍 默认地点：{self.default_loc}",
            f"🔄 缓存刷新：每30分钟自动",
            f"🌂 带伞提醒：降雨概率≥50%自动触发",
            f"⚠️ 极端预警：WMO极端天气码自动检测",
            f"",
            f"💬 【自然语言天气识别】",
            f"  状态：{nlp_status}",
            f"  回退地点：{nlp_fallback}",
            f"  关键词：天气、下雨、温度、台风等",
            f"  触发规则：私聊直接触发 / 群聊需@Bot",
            f"",
            f"⏰ 【天气提醒推送】",
            f"  状态：{reminder_status}",
            f"  时间：{reminder_cron}",
            f"  地点：{reminder_loc}",
            f"  目标：{len(reminder_targets)} 个会话",
        ]
        if reminder_targets:
            for t in reminder_targets[:5]:
                lines.append(f"    └ {t}")
            if len(reminder_targets) > 5:
                lines.append(f"    └ ... 还有 {len(reminder_targets)-5} 个")
        lines += [
            f"",
            f"📋 缓存地点：{', '.join(cache_keys[:10])}{'...' if len(cache_keys)>10 else ''}",
        ]
        yield event.chain_result([Plain("\n".join(lines))])

    # ── 自然语言天气意图识别 ──

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_nlp_weather(self, event: AstrMessageEvent):
        """监听所有消息，检测天气意图并自动回复

        行为规则：
        - 私聊：检测到天气关键词即触发
        - 群聊：需同时 @Bot 才触发
        - 命令（以/开头）不触发
        """
        # 检查功能开关
        nlp_enabled = self._get_cfg("nlp_weather_enabled", False)
        if not nlp_enabled:
            return

        text = event.message_str or ""
        text = text.strip()
        if not text:
            return

        # 跳过命令消息
        if text.startswith("/"):
            return

        # 天气关键词检测
        weather_kw = [
            "天气", "下雨", "雨", "温度", "降温", "升温", "气温",
            "台风", "暴雨", "雷暴", "刮风", "大风", "暴风",
            "多云", "晴", "阴天", "阴", "下雪", "雪", "雾",
        ]
        if not any(kw in text for kw in weather_kw):
            return

        # 群聊需 @Bot 才触发（私聊直接触发）
        if not event.is_private_chat():
            self_id = event.get_self_id()
            if not self_id:
                return
            mentioned = False
            for comp in event.get_messages():
                if hasattr(comp, 'qq') and str(comp.qq) == self_id:
                    mentioned = True
                    break
            if not mentioned:
                return

        # 从文本中提取地点
        location = _extract_location_nlp(text)
        if not location:
            location = self._get_cfg("nlp_weather_fallback", "广州")

        # 查询天气
        result = self._query_location(location)
        if not result:
            # 查询失败就不回复了，避免尴尬
            return

        self._nlp_query_count += 1

        loc_name = result["loc_name"]
        current = result["current"]
        daily = result["daily"]
        hourly_raw = result["hourly_raw"]

        report = _build_report(loc_name, current, daily, hourly_raw)
        await self.context.send_message(
            event.unified_msg_origin,
            MessageChain([Plain(report)])
        )
