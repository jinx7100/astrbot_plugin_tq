import os, re, json, datetime, time, requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as ImageW

import jieba

from astrbot.api.event import AstrMessageEvent, filter, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.event.filter import command
from astrbot.api.all import *
from astrbot.core.platform.message_session import MessageSession

GEO_API  = "https://geocoding-api.open-meteo.com/v1/search"
WEA_API  = "https://api.open-meteo.com/v1/forecast"

# ── 城市词库（用于 jieba NLP 地点提取） ──
CITY_LEXICON = {
"北京","上海","广州","深圳","杭州","成都","武汉","西安","南京","重庆","天津","苏州",
"长沙","郑州","东莞","青岛","沈阳","宁波","昆明","大连","厦门","合肥","佛山","福州",
"哈尔滨","济南","温州","长春","石家庄","常州","泉州","南宁","贵阳","南昌","太原",
"烟台","嘉兴","南通","金华","珠海","惠州","徐州","海口","乌鲁木齐","呼和浩特",
"银川","西宁","兰州","拉萨","桂林","洛阳","邯郸","威海","扬州","绍兴","保定",
"廊坊","唐山","秦皇岛","张家口","承德","沧州","衡水","邢台","大同","阳泉","长治",
"晋城","朔州","忻州","吕梁","晋中","临汾","运城","包头","乌海","赤峰","通辽",
"鄂尔多斯","呼伦贝尔","巴彦淖尔","乌兰察布","兴安盟","锡林郭勒","阿拉善","鞍山",
"抚顺","本溪","丹东","锦州","营口","阜新","辽阳","盘锦","铁岭","朝阳","葫芦岛",
"吉林","四平","辽源","通化","白山","松原","白城","延边","齐齐哈尔","鸡西","鹤岗",
"双鸭山","大庆","伊春","佳木斯","七台河","牡丹江","黑河","绥化","大兴安岭","无锡",
"连云港","淮安","盐城","镇江","泰州","宿迁","湖州","绍兴","金华","衢州","舟山",
"台州","丽水","芜湖","蚌埠","淮南","马鞍山","淮北","铜陵","安庆","黄山","滁州",
"阜阳","宿州","六安","亳州","池州","宣城","莆田","三明","漳州","南平","龙岩",
"宁德","景德镇","萍乡","九江","新余","鹰潭","赣州","吉安","宜春","抚州","上饶",
"淄博","枣庄","东营","潍坊","济宁","泰安","日照","临沂","德州","聊城","滨州",
"菏泽","开封","洛阳","平顶山","安阳","鹤壁","新乡","焦作","濮阳","许昌","漯河",
"三门峡","南阳","商丘","信阳","周口","驻马店","黄石","十堰","宜昌","襄阳","鄂州",
"荆门","孝感","荆州","黄冈","咸宁","随州","恩施","株洲","湘潭","衡阳","邵阳",
"岳阳","常德","张家界","益阳","郴州","永州","怀化","娄底","湘西","韶关","汕头",
"江门","湛江","茂名","肇庆","惠州","梅州","汕尾","河源","阳江","清远","中山",
"潮州","揭阳","云浮","柳州","桂林","梧州","北海","防城港","钦州","贵港","玉林",
"百色","贺州","河池","来宾","崇左","三亚","三沙","儋州","自贡","攀枝花","泸州",
"德阳","绵阳","广元","遂宁","内江","乐山","南充","眉山","宜宾","广安","达州",
"雅安","巴中","资阳","阿坝","甘孜","凉山","六盘水","遵义","安顺","毕节","铜仁",
"黔西南","黔东南","黔南","曲靖","玉溪","保山","昭通","丽江","普洱","临沧","楚雄",
"红河","文山","西双版纳","大理","德宏","怒江","迪庆","日喀则","昌都","林芝","山南",
"那曲","阿里","铜川","宝鸡","咸阳","渭南","延安","汉中","榆林","安康","商洛",
"嘉峪关","金昌","白银","天水","武威","张掖","平凉","酒泉","庆阳","定西","陇南",
"临夏","甘南","海东","海北","黄南","海南","果洛","玉树","海西","石嘴山","吴忠",
"固原","中卫","克拉玛依","吐鲁番","哈密","昌吉","博尔塔拉","巴音郭楞","阿克苏",
"克孜勒苏","喀什","和田","伊犁","塔城","阿勒泰","番禺","天河","越秀","海珠",
"荔湾","白云","黄埔","花都","南沙","从化","增城","福田","罗湖","南山","盐田",
"宝安","龙岗","龙华","坪山","光明","朝阳","海淀","丰台","西城","东城","通州",
"大兴","顺义","昌平","房山","浦东","徐汇","黄浦","静安","长宁","虹口","杨浦",
"普陀","闵行","宝山","松江","嘉定","青浦","奉贤","崇明","武侯","锦江","青羊",
"金牛","成华","高新","天府","雁塔","碑林","莲湖","未央","长安","江宁","鼓楼",
"玄武","秦淮","建邺","栖霞","雨花","吴中","姑苏","虎丘","吴江","萧山","余杭",
"西湖","拱墅","上城","滨江","临平","钱塘","洪山","武昌","江汉","江岸","硚口",
"汉阳","青山","东西湖","江夏","黄陂","新洲","岳麓","芙蓉","天心","开福","雨花",
"望城","禅城","南海","顺德","三水","高明","莞城","东城","南城","万江","长安",
"厚街","虎门","常平","塘厦","寮步","石龙","石碣","松山湖","中山","香洲","金湾",
"斗门","濠江","南澳","澄海","潮阳","潮南","揭东","普宁","陆丰","海丰","惠东",
"博罗","龙门","台山","开平","鹤山","恩平","四会","高要","广宁","德庆","封开",
"怀集","端州","鼎湖","源城","东源","和平","连平","龙川","紫金","梅县","兴宁",
"五华","丰顺","蕉岭","大埔","平远","乳源","乐昌","南雄","始兴","仁化","翁源",
"新丰","连州","英德","连山","连南","阳山","清新","佛冈","顺德","容桂","大良",
"伦教","勒流","北滘","陈村","乐从","均安","杏坛","龙江","九江","西樵","丹灶",
"狮山","大沥","里水","桂城","石岐","东区","西区","南区","五桂山","小榄","古镇",
"三角","民众","南朗","港口","大涌","沙溪","横栏","东升","东凤","南头","黄圃",
"三乡","板芙","神湾","坦洲","沙田","桥头","横沥","东坑","企石","石排","茶山",
"中堂","麻涌","望牛墩","道滘","洪梅","高埗","谢岗","清溪","凤岗","黄江","大朗"}
# 注册到 jieba 用户词典
for c in CITY_LEXICON:
    jieba.add_word(c)

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

# ── 缓存（带线程锁） ──
_cache = {}
_cache_lock = __import__('threading').Lock()
CACHE_TTL = 1800  # 30 分钟

def _cache_get(key):
    with _cache_lock:
        ent = _cache.get(key)
    if ent and time.time() - ent["ts"] < CACHE_TTL:
        return ent["data"]
    return None

def _cache_set(key, data):
    with _cache_lock:
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
        # 用 numpy polyfit 替代 scipy 样条插值
        deg = min(3, len(temps) - 1)
        coeffs = np.polyfit(x, y, deg)
        ys = np.polyval(coeffs, xs)
        # 中文字体探测：优先插件自带的 font.ttc，再回退系统和 matplotlib 内置
        fp = None
        for _f in [
            os.path.join(pdir, "font.ttc"),       # 插件自带的文泉驿正黑
            os.path.join(pdir, "fonts", "font.ttc"),
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]:
            if os.path.exists(_f):
                fp = _f
                break
        pr = fm.FontProperties(fname=fp, size=12) if fp else fm.FontProperties(size=12)
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

    使用 jieba 分词 + 城市词库匹配，优先提取已知地名。
    同时支持正则规则作为备用（处理未注册的地名组合）。
    """
    if not text:
        return None

    # ── Step 1: jieba 分词 + 词库匹配（准确度高） ──
    # 先去掉时间词，减少干扰
    cleaned = re.sub(r'(?:今天|明天|后天|大后天|昨天|前天|周末|下[周星]|这[周星]|下个月?|上个?月)', '', text).strip()
    words = list(jieba.cut(cleaned))

    # 按长度排序，优先返回最长匹配（"鄂尔多斯" > "鄂尔" 这种）
    found = [w for w in words if w in CITY_LEXICON]
    if found:
        found.sort(key=len, reverse=True)
        return found[0]

    # ── Step 2: 正则备用（处理未录入词库的组合） ──
    # 模式: "X天气"、"X下雨" 等
    m = re.search(r'([\u4e00-\u9fff]{2,5})(?:的)?(?:天气|下雨|下雪|温度|气温|台风|大风|暴雨|雷暴|雾|雪|晴|阴)', cleaned)
    if m:
        return m.group(1)
    # 模式: "在X"、"去X"、"到X"
    m = re.search(r'(?:在|到|去|回|上|下)([\u4e00-\u9fff]{2,5})(?:的)?(?:天气|下雨|温度|那|这|如何|怎样|怎么样)?', cleaned)
    if m:
        return m.group(1)
    # 模式: "X那边"、"X这里"
    m = re.search(r'([\u4e00-\u9fff]{2,5})(?:那边|这里|那里|这边)', cleaned)
    if m:
        return m.group(1)

    return None


def _build_hourly_dict(hourly_raw):
    """从原始逐时数据构建 hourly_dict（抽取为公共函数，消除重复）"""
    if not hourly_raw:
        return []
    hourly_dict = []
    for t, tmp, cd, pr in zip(
        hourly_raw["time"], hourly_raw["temperature_2m"],
        hourly_raw["weathercode"],
        hourly_raw.get("precipitation_probability", [0] * len(hourly_raw["time"]))
    ):
        w = WMO.get(cd, ("未知", "not_supported", 0, 0))
        hourly_dict.append({"t": t, "tmp": tmp, "icon": w[1]})
    return hourly_dict


# ══════════════════════════════════════
#  插件主类
# ══════════════════════════════════════

@register("天枢", "理予", "查天气、3天预报、极端预警、自动带伞提醒、自然语言天气识别", "2.3.0", "")
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
        hourly_dict = _build_hourly_dict(hourly_raw)
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
            # 先尝试直接用参数查，查不到再用 NLP 提取
            result = self._query_location(q)
            if not result:
                # 尝试从自然语言中提取地点
                nlp_loc = _extract_location_nlp(q)
                if nlp_loc:
                    q = nlp_loc
                    result = self._query_location(q)
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

        hourly_dict = _build_hourly_dict(hourly_raw)
        img = None
        if hourly_dict:
            img = _gen_chart(loc_name, hourly_dict, event.unified_msg_origin, pdir)

        if img:
            yield event.chain_result([Plain(report + "\n"), Image.fromFileSystem(img)])
            try: os.remove(img)
            except: pass
        else:
            yield event.chain_result([Plain(report + "\n\n（📊 温度图表暂时生成失败，不影响文字预报）")])

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
            f"📊 【天枢 - 运行状态】",
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
            await self.context.send_message(
                event.unified_msg_origin,
                MessageChain([Plain(f"唔…理予没查到「{location}」的天气，要不换个名字试试？")])
            )
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
