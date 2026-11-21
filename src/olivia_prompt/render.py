"""Three-tier renderer.

The renderer is the reason the prompt format can never drift: the model never
writes the final string, it only fills slots.  Everything below is deterministic
— the same :class:`SlotSet` always produces byte-identical output, on every
platform, forever.  That is what lets users edit a template and re-render
instead of re-running the model.

Tiers
-----
short   40-80 tokens, pure booru tag string
medium  ~150 words, three paragraphs of prose
long    ~230-380 words, four paragraphs / fourteen sentence positions
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from .ecosystem import Ecosystem, get_ecosystem
from .slots import TIERS, SlotSet

# --------------------------------------------------------------------------- #
# Pose grammar tables
# --------------------------------------------------------------------------- #

#: pose family -> (verb, preposition, object-rest clause, shoulder clause, ground clause)
#: The shoulder clause deliberately starts with the body part, not a pronoun, so
#: the caller can prefix it with the right possessive exactly once.
_POSE: Dict[str, Tuple[str, str, str, str, str]] = {
    "sit": (
        "sits", "on",
        "resting on {p} lap",
        "shoulders are slightly inclined forward",
        "both feet rest naturally on the {g}",
    ),
    "stand": (
        "stands", "at",
        "held loosely in both hands",
        "shoulders are level and relaxed",
        "both feet are planted evenly on the {g}",
    ),
    "walk": (
        "walks", "through",
        "carried at {p} side",
        "shoulders are slightly inclined forward",
        "both feet meet the {g} in mid-stride",
    ),
    "lean": (
        "leans", "against",
        "held against {p} chest",
        "shoulders are angled into the support",
        "both feet rest naturally on the {g}",
    ),
    "kneel": (
        "kneels", "on",
        "resting on {p} thigh",
        "spine is held upright",
        "one knee rests on the {g}",
    ),
    "lie": (
        "lies", "on",
        "resting on {p} chest",
        "shoulders are relaxed against the {g}",
        "{p} body rests along the {g}",
    ),
    "crouch": (
        "crouches", "in",
        "held close to {p} knees",
        "spine is curled forward",
        "both feet are flat on the {g}",
    ),
}

_GENDER_NOUN = {"female": "woman", "male": "man", "other": "person"}
_COUNT_PHRASE = {"1": "a single", "2": "two", "3": "three"}
_COUNT_TAG = {
    "female": ("1girl", "2girls", "3girls"),
    "male": ("1boy", "2boys", "3boys"),
    "other": ("1other", "2others", "3others"),
}

_COLOR_TAGS = {
    "navy": "navy", "black": "black", "white": "white", "red": "red",
    "blue": "blue", "green": "green", "ivory": "white", "cream": "cream",
    "grey": "grey", "gray": "grey", "brown": "brown", "pink": "pink",
    "purple": "purple", "gold": "gold", "silver": "silver", "amber": "orange",
    "teal": "teal", "beige": "beige",
}

_GARMENT_TAGS = {
    "dress", "skirt", "blouse", "cardigan", "coat", "jacket", "hoodie",
    "jeans", "shirt", "sweater", "robe", "kimono", "hanfu", "camisole",
    "trousers", "shorts", "vest", "blazer", "cloak", "cape", "uniform",
    "turtleneck",
}

_POSE_TAGS = {
    "sit": "sitting", "stand": "standing", "walk": "walking",
    "lean": "leaning_forward", "kneel": "kneeling", "lie": "lying",
    "crouch": "crouching",
}

_SETTING_TAGS = {
    "garden": "garden", "rooftop": "rooftop", "library": "library",
    "alley": "alley", "street": "street", "room": "indoors", "cafe": "cafe",
    "forest": "forest", "beach": "beach", "shrine": "shrine", "station": "station",
    "balcony": "balcony", "classroom": "classroom", "kitchen": "kitchen",
    "temple": "temple", "ruins": "ruins", "train": "train_interior",
    "roof": "rooftop", "courtyard": "courtyard", "bar": "bar",
}

_LIGHT_TAGS = (
    ("late-afternoon", "afternoon"),
    ("late afternoon", "afternoon"),
    ("golden hour", "golden_hour"),
    ("sunset", "sunset"),
    ("sunrise", "sunrise"),
    ("dusk", "dusk"),
    ("midnight", "night"),
    ("night", "night"),
    ("overcast", "overcast"),
    ("moonlight", "moonlight"),
    ("morning", "morning"),
    ("afternoon", "afternoon"),
)

_DIR_TAGS = {
    "left": "from_left", "right": "from_right", "front": "front_lighting",
    "top": "top_lighting", "back-right": "backlighting", "back-left": "backlighting",
}

_MEDIUM_NOUN = {
    "anime illustration": "illustration",
    "illustration": "illustration",
    "cinematic photograph": "photograph",
    "cinematic photography": "photograph",
    "photography": "photograph",
    "oil painting": "painting",
    "painting": "painting",
    "3d render": "render",
    "render": "render",
    "watercolor": "watercolor",
    "pixel art": "pixel art",
}

_MEDIUM_TAG = {
    "anime illustration": "illustration",
    "illustration": "illustration",
    "cinematic photography": "photorealistic",
    "photography": "photorealistic",
    "oil painting": "painting",
    "painting": "painting",
    "3d render": "3d",
    "render": "3d",
    "watercolor": "watercolor",
    "pixel art": "pixel_art",
}

#: Mediums that are mass nouns and must not take an article in prose.
_MASS_MEDIUM = ("photography", "watercolor", "pixel art")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _cap(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _lc(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def _medium_phrase(medium: str) -> str:
    """Turn a style token into a count noun usable mid-sentence."""
    m = (medium or "").strip()
    low = m.lower()
    if low.endswith("photography"):
        return re.sub(r"photography$", "photograph", m, flags=re.I)
    if low in ("watercolor", "pixel art"):
        return m
    return m


def _medium_noun(medium: str) -> str:
    low = (medium or "").strip().lower()
    if low in _MEDIUM_NOUN:
        return _MEDIUM_NOUN[low]
    if low.endswith("photography"):
        return "photograph"
    words = (medium or "").strip().split()
    return words[-1] if words else "illustration"


def _medium_tag(medium: str) -> str:
    low = (medium or "").strip().lower()
    if low in _MEDIUM_TAG:
        return _MEDIUM_TAG[low]
    words = re.findall(r"[a-z]+", low)
    return words[-1] if words else "illustration"


def _fix_pronoun(text: str, ss: SlotSet) -> str:
    """Rewrite third-person pronouns inside literal slot text to match gender."""
    if not text:
        return text
    poss = ss.pronoun("poss")
    out = re.sub(r"\bher\b", poss, text)
    out = re.sub(r"\bhis\b", poss, out)
    out = re.sub(r"\btheir\b", poss, out)
    return out


def _with_article(text: str) -> str:
    """Prefix a noun phrase with a/an unless it already has one or is plural."""
    t = (text or "").strip()
    if not t:
        return t
    low = t.lower()
    if low.startswith(("a ", "an ", "the ", "some ", "his ", "her ", "their ")):
        return t
    last = low.split()[-1]
    if last.endswith("s") and not last.endswith(("ss", "us", "is")):
        return t  # plural count noun: no article
    article = "an" if low[0] in "aeiou" else "a"
    return f"{article} {t}"


def _with_detail(main: str, detail: str) -> str:
    """'deep navy dress' + 'delicate ivory embroidery' -> 'a deep navy dress with ...'"""
    main = (main or "").strip()
    if not main:
        return ""
    article = "an" if main[0].lower() in "aeiou" else "a"
    if detail:
        return f"{article} {main} with {detail}"
    return f"{article} {main}"


def _join(items: Iterable[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _strip_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", (text or "").strip(), flags=re.I)


_NO_SUN = ("night", "midnight", "moon", "dusk", "evening", "overcast", "indoor", "candle", "neon")


def _light_word(light_time: str) -> str:
    """'sunlight' is wrong for a midnight or overcast scene."""
    low = (light_time or "").lower()
    return "light" if any(n in low for n in _NO_SUN) else "sunlight"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _quality_line(ss: SlotSet) -> str:
    parts: List[str] = [q for q in (ss.get("quality") or ()) if q]
    if ss.get("medium"):
        parts.append(ss["medium"])
    if ss.get("model_style"):
        parts.append(ss["model_style"])
    return ", ".join(parts) + "," if parts else ""


# --------------------------------------------------------------------------- #
# long tier — 14 sentence positions
# --------------------------------------------------------------------------- #

def _long(ss: SlotSet) -> str:
    poss = ss.pronoun("poss")
    subj = _cap(ss.pronoun("subj"))
    pose = ss["pose_family"]
    verb, prep, rest, shoulder, ground_clause = _POSE[pose]
    ground = _fix_pronoun(ss["ground"] or ss["setting"], ss)
    anchors = {"p": poss, "g": ground}

    blocks: List[str] = []

    prefix = _quality_line(ss)
    if prefix:
        blocks.append(prefix)

    # P1a — subject anchor
    count = _COUNT_PHRASE.get(str(ss["subject_count"]), "a single")
    noun = _GENDER_NOUN.get(ss["gender"], "person")
    p1a = (
        f"A highly detailed {_medium_phrase(ss['medium'])} of {count} {ss['skin']} "
        f"{ss['age']} {noun} with {ss['hair_length']} {ss['hair_color']} hair, "
        f"{ss['eye_color']} eyes and a {ss['expression']} expression."
    )
    # P1b — wardrobe chain
    wardrobe = _with_detail(ss["outfit_main"], ss["outfit_detail"])
    extras = [_with_article(x) for x in (ss["outfit_layer"], ss["bartwear"]) if x]
    p1b = f"{subj} wears {wardrobe}, {_join(extras)}." if extras else f"{subj} wears {wardrobe}."
    acc = [a for a in (ss.get("accessories") or ()) if a]
    if acc:
        p1b = p1b[:-1] + f", finished with {_join(acc)}."
    # P1c — micro-detail + judgement clause
    craft = [c for c in (ss.get("craft") or ()) if c]
    while len(craft) < 2:
        craft.append("softly folded fabric")
    p1c = (
        f"Fine strands of hair, {craft[0]} and {craft[1]} give {poss} appearance "
        f"a natural, carefully observed quality."
    )
    blocks.append(" ".join((p1a, p1b, p1c)))

    # P2 — pose, action, physical plausibility
    action = _fix_pronoun(ss["action"], ss)
    p2a = (
        f"{subj} {verb} {prep} a {ss['perch']} in a {ss['setting']}, "
        f"{action} {rest.format(**anchors)}, one hand supports the {ss['hand_support']} "
        f"while the other {ss['hand_other']} with a {ss['grip']}."
    )
    gaze = _fix_pronoun(ss["gaze_target"], ss)
    p2b = (
        f"{_cap(poss)} gaze follows {gaze}, {poss} {shoulder.format(**anchors)}, "
        f"and {ground_clause.format(**anchors)}."
    )
    vis = [v for v in (ss.get("visibility") or ()) if v]
    while len(vis) < 3:
        vis.append("hands")
    p2c = f"Keep {poss} {vis[0]}, {vis[1]} and {vis[2]} clearly visible."
    blocks.append(" ".join((p2a, p2b, p2c)))

    # P3 — world, materials, detail falloff
    props = [p for p in (ss.get("props") or ()) if p]
    while len(props) < 3:
        props.append("low planters")
    p3a = f"{_cap(props[0])}, {props[1]} and {props[2]} frame the {ss['anchor']}."
    p3b = f"A {ss['opening']} opens onto layers of {ss['background']}."
    mats = [m for m in (ss.get("materials") or ()) if m]
    while len(mats) < 3:
        mats.append("stone")
    p3c = (
        f"Show subtle wear on the {mats[0]}, fine veins in {mats[1]} and variations "
        f"in the {mats[2]} surface, with progressively softer detail toward the "
        f"distant {ss['bg_far']}."
    )
    blocks.append(" ".join((p3a, p3b, p3c)))

    # P4 — light, palette, composition, render judgement
    hits = [_fix_pronoun(h, ss) for h in (ss.get("hit") or ()) if h]
    while len(hits) < 2:
        hits.append("the fabric")
    p4a = (
        f"{ss['light_time']} {_light_word(ss['light_time'])} enters from the "
        f"{ss['light_dir']}, catching the {hits[0]} and the {hits[1]}."
    )
    p4b = f"Soft {ss['cool_color']} shadows balance the {ss['warm_color']} highlights."
    p4c = (
        f"Use a {ss['aspect']} composition at {ss['cam_height']}, a readable "
        f"character silhouette, controlling depth and restrained background detail."
    )
    renders = [r for r in (ss.get("render") or ()) if r]
    while len(renders) < 3:
        renders.append("distinct material textures")
    p4d = (
        f"{renders[0]}, {renders[1]} and {renders[2]} create a {ss['mood']} "
        f"{_medium_noun(ss['medium'])}."
    )
    blocks.append(" ".join((p4a, p4b, p4c, p4d)))

    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# medium tier — three paragraphs
# --------------------------------------------------------------------------- #

def _medium(ss: SlotSet) -> str:
    poss = ss.pronoun("poss")
    subj = _cap(ss.pronoun("subj"))
    verb, prep, _rest, shoulder, ground_clause = _POSE[ss["pose_family"]]
    ground = _fix_pronoun(ss["ground"] or ss["setting"], ss)
    anchors = {"p": poss, "g": ground}

    count = _COUNT_PHRASE.get(str(ss["subject_count"]), "a single")
    noun = _GENDER_NOUN.get(ss["gender"], "person")
    wardrobe = _with_detail(ss["outfit_main"], ss["outfit_detail"])
    extras = [_with_article(x) for x in (ss["outfit_layer"], ss["bartwear"]) if x]
    head = _with_article(_medium_phrase(ss["medium"]))
    p1 = (
        f"{_cap(head)} of {count} {ss['skin']} {ss['age']} {noun} with "
        f"{ss['hair_length']} {ss['hair_color']} hair, {ss['eye_color']} eyes and "
        f"a {ss['expression']} expression, wearing {wardrobe}"
        f"{', ' + _join(extras) if extras else ''}."
    )

    gaze = _fix_pronoun(ss["gaze_target"], ss)
    p2 = (
        f"{subj} {verb} {prep} a {ss['perch']} in a {ss['setting']}, "
        f"{_fix_pronoun(ss['action'], ss)}. {_cap(poss)} gaze follows {gaze}, "
        f"{poss} {shoulder.format(**anchors)}, and {ground_clause.format(**anchors)}."
    )

    hits = [_fix_pronoun(h, ss) for h in (ss.get("hit") or ()) if h]
    hit = hits[0] if hits else "the silhouette"
    props = [p for p in (ss.get("props") or ()) if p]
    renders = [r for r in (ss.get("render") or ()) if r]
    r1 = _lc(renders[0]) if renders else "refined rendering"
    r2 = _lc(renders[1]) if len(renders) > 1 else "clean material separation"
    p3 = (
        f"{ss['light_time']} {_light_word(ss['light_time'])} enters from the "
        f"{ss['light_dir']}, catching {hit}. "
        f"{_cap(_join(props[:2])) or 'Nearby details'} frame the {ss['anchor']}, "
        f"opening onto {ss['background']}. Use a {ss['aspect']} composition at "
        f"{ss['cam_height']}; {r1} and {r2} create a {ss['mood']} "
        f"{_medium_noun(ss['medium'])}."
    )

    blocks = [p1, p2, p3]
    prefix = _quality_line(ss)
    if prefix:
        blocks.insert(0, prefix)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# short tier — booru tags
# --------------------------------------------------------------------------- #

def _tag_subject(ss: SlotSet) -> List[str]:
    table = _COUNT_TAG.get(str(ss["gender"]), _COUNT_TAG["other"])
    idx = {"1": 0, "2": 1, "3": 2}.get(str(ss["subject_count"]), 0)
    tags = [table[idx]]
    if idx == 0:
        tags.append("solo")
    return tags


def _tag_wardrobe(ss: SlotSet) -> List[str]:
    tags: List[str] = []
    words = re.findall(r"[a-z]+", str(ss["outfit_main"]).lower())
    color = next((_COLOR_TAGS[w] for w in words if w in _COLOR_TAGS), None)
    garment = next((w for w in words if w in _GARMENT_TAGS), None)
    if garment and color:
        tags.append(f"{color}_{garment}")
    elif garment:
        tags.append(garment)
    elif color:
        tags.append(f"{color}_outfit")
    for slot in ("outfit_layer", "bartwear"):
        for w in re.findall(r"[a-z]+", str(ss[slot]).lower()):
            if w in _GARMENT_TAGS:
                tags.append(w)
                break
    for a in (ss.get("accessories") or ()):
        slug = _slug(_strip_article(a))
        if slug:
            tags.append(slug)
    return tags


def _tag_setting(ss: SlotSet) -> List[str]:
    words = re.findall(r"[a-z]+", str(ss["setting"]).lower())
    tags = [_SETTING_TAGS[w] for w in words if w in _SETTING_TAGS]
    if ss["indoor"] and "indoors" not in tags:
        tags.append("indoors")
    return tags


def _tag_light(ss: SlotSet) -> List[str]:
    tags: List[str] = []
    low = str(ss["light_time"]).lower()
    for needle, tag in _LIGHT_TAGS:
        if needle in low:
            tags.append(tag)
            break
    tags.append(_DIR_TAGS.get(str(ss["light_dir"]), "from_left"))
    return tags


def _tag_camera(ss: SlotSet) -> List[str]:
    tags: List[str] = ["eye_level"]
    aspect = str(ss["aspect"]).lower()
    if "horizontal" in aspect or "landscape" in aspect:
        tags.append("wide_shot")
    framing = str(ss.get("framing") or "").lower()
    if "close" in framing:
        tags.append("close-up")
    return tags


def _short(ss: SlotSet) -> str:
    tags: List[str] = []
    tags.extend(q for q in (ss.get("quality") or ()) if q)
    tags.append(_medium_tag(ss["medium"]))
    tags.extend(_tag_subject(ss))
    tags.append(f"{_slug(ss['hair_length'])}_hair")
    tags.append(f"{_slug(ss['hair_color'])}_hair")
    eye = re.split(r"[-,/ ]", str(ss["eye_color"]).lower())[0]
    if eye:
        tags.append(f"{_slug(eye)}_eyes")
    expr = re.split(r"[,/]", str(ss["expression"]).lower())[0].strip()
    if expr:
        tags.append(_slug(expr))
    tags.extend(_tag_wardrobe(ss))
    tags.append(_POSE_TAGS.get(str(ss["pose_family"]), "sitting"))
    tags.extend(_tag_camera(ss))
    tags.extend(_tag_setting(ss))
    tags.extend(_tag_light(ss))
    for extra in (ss.get("tags_extra") or ()):
        slug = _slug(extra)
        if slug:
            tags.append(slug)

    seen = set()
    out: List[str] = []
    for t in tags:
        t = t.replace("__", "_").strip("_")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def render(slots: SlotSet, tier: str = "long", ecosystem: str | Ecosystem | None = None) -> str:
    """Render one tier of the positive prompt."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {TIERS}")
    eco = ecosystem if isinstance(ecosystem, Ecosystem) else get_ecosystem(ecosystem)
    if tier == "short":
        text = _short(slots)
        if eco.tag_style == "mj":
            text = text.replace("_", " ")
        return text
    if tier == "medium":
        return _medium(slots)
    return _long(slots)


def render_all(
    slots: SlotSet,
    ecosystem: str | Ecosystem | None = None,
    tiers: Sequence[str] = TIERS,
) -> Dict[str, str]:
    """Render several tiers in one pass."""
    eco = ecosystem if isinstance(ecosystem, Ecosystem) else get_ecosystem(ecosystem)
    return {t: render(slots, t, eco) for t in tiers}


__all__ = ["render", "render_all"]