"""Persona / wardrobe / style pools for the rule-based sampler.

Every entry carries a Chinese gloss because the same pools feed the M4 intent
synthesizer: a sample must be able to *say in Chinese* exactly what it decided,
so provenance (``user`` vs ``inferred``) stays honest in the training targets.

Pools are deliberately small-but-crossed: 9 hair colors x 14 outfits x 20
activities x 60 scenes already exceed 150k distinct worlds, while each pool
stays inspectable and editable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---- identity --------------------------------------------------------------
# (hair_en, hair_zh, eye_en) — eye is chosen with the hair, not independently,
# because "silver hair + amber eyes" is a look, not two coin flips.
PERSONAS: Tuple[Tuple[str, str, str], ...] = (
    ("silver", "银发", "blue-gray"),
    ("black", "黑发", "amber"),
    ("chestnut brown", "栗棕发", "hazel"),
    ("ash blonde", "灰金发", "pale green"),
    ("dark green", "墨绿发", "hazel"),
    ("deep blue", "深蓝发", "gray"),
    ("wine red", "酒红发", "crimson"),
    ("ivory white", "象牙白发", "pale violet"),
    ("flaxen gold", "亚麻金发", "amber"),
    ("auburn", "赤褐发", "green"),
    ("light brown", "浅棕发", "brown"),
    ("platinum blonde", "白金发", "ice blue"),
    ("teal", "蓝绿发", "teal"),
    ("lavender", "淡紫发", "violet"),
    ("honey blonde", "蜜金发", "light brown"),
    ("copper", "铜红发", "amber"),
)

SKINS: Tuple[Tuple[str, str], ...] = (
    ("fair-skinned", "白皙"),
    ("tan", "小麦色"),
    ("olive-skinned", "橄榄色"),
    ("pale-skinned", "苍白"),
    ("sun-kissed", "健康古铜"),
    ("porcelain-skinned", "白瓷"),
)

AGES: Tuple[Tuple[str, str], ...] = (
    ("adult", ""),
    ("young adult", "年轻"),
    ("middle-aged", "中年"),
    ("elderly", "年长"),
)

EXPRESSIONS: Tuple[Tuple[str, str], ...] = (
    ("calm, attentive", "神情安静而专注"),
    ("focused, quiet", "神色专注而安静"),
    ("joyful, relaxed", "神色明快而放松"),
    ("guarded, tired", "带着谨慎的疲惫"),
    ("serene, thoughtful", "安宁而若有所思"),
    ("curious, bright", "好奇而明亮"),
    ("wry, patient", "淡然而耐心"),
    ("soft, dreamy", "神情柔软而恍惚"),
    ("tender, wistful", "温柔而怅然"),
    ("sharp, confident", "锐利而自信"),
    ("sleepy, content", "睡意惺忪而满足"),
    ("earnest, absorbed", "认真而投入"),
)

HAIR_LENGTHS: Tuple[Tuple[str, str], ...] = (
    ("long", "长发"),
    ("very long", "及腰长发"),
    ("medium", "及肩发"),
    ("short", "短发"),
)

# ---- wardrobe --------------------------------------------------------------
# (outfit_main, zh) — outfit_main must carry a color word + a garment word so
# the short-tier tag extractor finds both.
OUTFITS: Tuple[Tuple[str, str], ...] = (
    ("deep navy dress", "深蓝连衣裙"),
    ("ivory linen blouse", "象牙白亚麻衬衫"),
    ("charcoal wool coat", "炭灰呢子大衣"),
    ("olive canvas jacket", "橄榄绿帆布夹克"),
    ("washed denim overalls", "水洗牛仔背带裤"),
    ("black turtleneck sweater", "黑色高领毛衣"),
    ("cream knit cardigan", "奶油色针织开衫"),
    ("burgundy pleated skirt", "酒红百褶裙"),
    ("forest-green qipao", "墨绿旗袍"),
    ("dusty-blue work apron", "蓝灰工作围裙"),
    ("mustard corduroy trousers", "芥末黄灯芯绒长裤"),
    ("slate-grey blazer", "石板灰西装外套"),
    ("rust-brown leather apron", "铁锈棕皮围裙"),
    ("white chef jacket", "白色厨师服"),
    ("sage-green linen dress", "灰绿亚麻连衣裙"),
    ("dark plum hoodie", "深梅紫连帽衫"),
    ("camel duffle coat", "驼色牛角扣大衣"),
    ("ink-black wide trousers", "墨黑阔腿长裤"),
    ("crimson silk blouse", "绯红丝绸衬衫"),
    ("pale-blue work shirt", "浅蓝工装衬衫"),
    ("moss-green parka", "苔绿派克大衣"),
    ("tan suede jacket", "棕黄麂皮夹克"),
    ("grey sailor-collar dress", "灰色水手领连衣裙"),
    ("wine-red knitted vest", "酒红针织马甲"),
)

OUTFIT_DETAILS: Tuple[Tuple[str, str], ...] = (
    ("delicate ivory embroidery", "细巧的象牙白刺绣"),
    ("worn stitching at the cuffs", "袖口磨旧的针脚"),
    ("faint woven stripe", "若隐若现的织纹"),
    ("mended patch at the elbow", "肘部缝补过的补丁"),
    ("hand-rolled hem", "手工卷边"),
    ("subtle plaid weave", "细小的格纹"),
    ("brass buttons worn dull", "磨暗的黄铜纽扣"),
    ("a drawstring pulled loose", "松开的抽绳"),
    ("faint bleach marks near the hem", "下摆附近的淡淡漂痕"),
    ("double-stitched shoulders", "双线缝制的肩线"),
    ("a thin leather belt", "一条细皮带"),
    ("small fabric-covered buttons", "包布小纽扣"),
)

OUTFIT_LAYERS: Tuple[Tuple[str, str], ...] = (
    ("lightweight cream cardigan", "轻薄奶油色开衫"),
    ("grey ribbed turtleneck", "灰色罗纹高领"),
    ("washed linen scarf", "洗旧亚麻围巾"),
    ("slate knitted shawl", "石板灰针织披肩"),
    ("olive field vest", "橄榄绿工装马甲"),
    ("cropped denim jacket", "短款牛仔外套"),
    ("wool pinafore layer", "羊毛背带层"),
    ("plaid flannel overshirt", "格纹法兰绒罩衫"),
    ("crocheted shawl", "钩织披肩"),
    ("rain-darkened poncho", "被雨打深的斗篷"),
)

BARTWEARS: Tuple[Tuple[str, str], ...] = (
    ("simple leather ankle boots", "简约皮质短靴"),
    ("scuffed combat boots", "磨损的作战靴"),
    ("woven straw sandals", "编织草鞋"),
    ("polished oxford shoes", "擦亮的牛津鞋"),
    ("canvas high-tops", "帆布高帮鞋"),
    ("wool-lined winter boots", "羊毛内衬冬靴"),
    ("low block-heel shoes", "低粗跟皮鞋"),
    ("felted wool slippers", "毡化羊毛居家鞋"),
    ("rubber work clogs", "橡胶工作木屐"),
    ("two-tone saddle shoes", "拼色马鞍鞋"),
)

ACCESSORIES: Tuple[Tuple[str, str], ...] = (
    ("", ""),
    ("a thin silver chain", "一条细银链"),
    ("a cloth wristband", "一条布腕带"),
    ("round wire glasses", "圆框细丝眼镜"),
    ("small pearl earrings", "小巧珍珠耳钉"),
    ("a woven straw hat", "一顶编织草帽"),
    ("a knitted wool scarf", "一条针织羊毛围巾"),
    ("a leather tool belt", "一条皮质工具腰带"),
    ("drop earrings of green glass", "绿玻璃坠式耳环"),
)

CRAFTS: Tuple[str, ...] = (
    "stitched hems",
    "softly folded fabric",
    "well-oiled leather",
    "hand-worn handles",
    "softly frayed cuffs",
    "faintly rumpled sleeves",
    "patinated buckles",
    "brush-worn fingertips",
    "dust-flecked knuckles",
    "sun-faded shoulder seams",
    "snagged thread loops",
    "faint wool pilling",
)

GRIPS: Tuple[str, ...] = (
    "relaxed, believable grip",
    "practised, unhurried grip",
    "light, careful grip",
    "steady, deliberate grip",
    "confident, weathered grip",
    "tentative, precise grip",
    "loose, easy grip",
    "firm, work-worn grip",
)

# ---- camera / aspect -------------------------------------------------------
# (aspect, zh) — zh doubles as the "留白方向" phrase in long intents.
ASPECTS: Tuple[Tuple[str, str], ...] = (
    ("spacious horizontal", "横向留白，主体两侧留出呼吸感"),
    ("tall vertical", "纵向留白，主体上方留出空间"),
    ("square intimate", "方形构图，主体紧凑居中"),
    ("wide panoramic", "开阔全景，主体置于画面一端"),
)

#: pose_family -> (cam_height en, cam zh)
CAM_HEIGHTS: Dict[str, Tuple[str, str]] = {
    "sit": ("seated eye level", "坐姿视平线机位"),
    "kneel": ("low kneeling height", "低跪高度机位"),
    "crouch": ("low crouch height", "低蹲高度机位"),
    "lie": ("low, at ground level", "贴近地面的低机位"),
    "lean": ("standing eye level", "站姿视平线机位"),
    "stand": ("standing eye level", "站姿视平线机位"),
    "walk": ("standing eye level", "站姿视平线机位"),
}

LIGHT_DIR_ZH: Dict[str, str] = {
    "left": "左侧",
    "right": "右侧",
    "front": "正面",
    "top": "头顶",
    "back-left": "左后方",
    "back-right": "右后方",
}

# ---- light palette ---------------------------------------------------------
# phase keyword -> (cool, warm); detected against the scene's light_time text.
LIGHT_PALETTES: Tuple[Tuple[Tuple[str, ...], Tuple[str, str]], ...] = (
    (("morning", "dawn"), ("soft blue", "pale gold")),
    (("noon", "midday"), ("neutral slate", "white-gold")),
    (("afternoon",), ("dusty blue", "honey")),
    (("golden hour", "golden dawn", "sunset"), ("violet", "honey amber")),
    (("dusk", "evening", "tungsten"), ("indigo", "tungsten orange")),
    (("blue hour", "pre-dawn"), ("cold cyan", "ember orange")),
    (("night", "moon", "candle", "neon", "aurora"), ("cold cyan", "ember orange")),
    (("overcast",), ("slate gray", "muted silver")),
    (("underwater",), ("deep teal", "shaft gold")),
    (("winter",), ("icy blue", "pale cream")),
)

# ---- medium / render -------------------------------------------------------
# (medium, weight, render triple)
MEDIUMS: Tuple[Tuple[str, int, Tuple[str, str, str]], ...] = (
    ("anime illustration", 45, ("Refined linework", "nuanced skin shading", "distinct material textures")),
    ("cinematic photography", 20, ("Anamorphic lens flare", "grainy high-ISO texture", "volumetric haze")),
    ("oil painting", 10, ("Visible brushwork", "impasto highlights", "soft glaze transitions")),
    ("watercolor", 15, ("Granulating pigment washes", "blooming wet edges", "spare ink linework")),
    ("3d render", 10, ("Physically-based shading", "soft global illumination", "crisp specular falloff")),
)

MODEL_STYLES: Tuple[str, ...] = ("", "", "", "krea2 style")

QUALITY: Tuple[str, str] = ("masterpiece", "very aesthetic")


def light_palette(light_time: str) -> Tuple[str, str]:
    low = (light_time or "").lower()
    for needles, (cool, warm) in LIGHT_PALETTES:
        if any(n in low for n in needles):
            return cool, warm
    return "dusty blue", "honey"


def pick_weighted(rng, items: List[Tuple[str, int]]) -> str:
    """Weighted pick over (value, weight) pairs."""
    total = sum(w for _, w in items)
    roll = rng.randrange(total)
    upto = 0
    for value, w in items:
        upto += w
        if roll < upto:
            return value
    return items[-1][0]


__all__ = [
    "PERSONAS",
    "SKINS",
    "AGES",
    "EXPRESSIONS",
    "HAIR_LENGTHS",
    "OUTFITS",
    "OUTFIT_DETAILS",
    "OUTFIT_LAYERS",
    "BARTWEARS",
    "ACCESSORIES",
    "CRAFTS",
    "GRIPS",
    "ASPECTS",
    "CAM_HEIGHTS",
    "LIGHT_DIR_ZH",
    "LIGHT_PALETTES",
    "MEDIUMS",
    "MODEL_STYLES",
    "QUALITY",
    "light_palette",
    "pick_weighted",
]
