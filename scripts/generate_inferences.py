"""Generate the 228-prompt DEFAULT inference set: 57 themes × 4 languages
(en/es/ar/zh) — 46 base + 1 multi-turn + 10 special (5 tools + 5 response_format).

Each theme is preserved semantically across languages. Outputs to
`inferences/default/<theme>_<lang>.json` with shape `{messages, max_tokens, seed}`.
This is the set `e2e infer` runs by default. Other sets (e.g.
`inferences/kimi-specific/`) live alongside it and are selected with
`e2e infer --inferences-dir inferences/<set>`.

Run from repo root: `python3 scripts/generate_inferences.py`
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "inferences" / "default"
OUT.mkdir(parents=True, exist_ok=True)

# Seed convention: theme_idx*100 + lang_offset (en=0, es=1, ar=2, zh=3)
LANG_OFFSET = {"en": 0, "es": 1, "ar": 2, "zh": 3}


# ─────────────────────────────────────────────────────────────────────────
# Each theme: (theme_name, max_tokens, base_seed, {lang: {system, user}})
# When system is empty string, only user message is included.
# Multi-turn themes use {messages: [...]} directly.
# ─────────────────────────────────────────────────────────────────────────

THEMES = [

# ── Math / Reasoning ─────────────────────────────────────────────────────
("math_arithmetic", 512, 1001, {
    "en": {"system": "You are a careful step-by-step math tutor. Always show working.",
           "user": "Solve: ((127 * 53) - (89 * 47)) / 13. Explain each step."},
    "es": {"system": "Eres un tutor de matemáticas paciente que muestra cada paso del razonamiento.",
           "user": "Resuelve: ((127 × 53) − (89 × 47)) ÷ 13. Explica cada paso."},
    "ar": {"system": "أنت معلم رياضيات صبور يعرض كل خطوة من خطوات الحل بوضوح.",
           "user": "احسب: ((127 × 53) − (89 × 47)) ÷ 13. اشرح كل خطوة بالتفصيل."},
    "zh": {"system": "你是一位耐心的数学老师,总是逐步展示解题过程。",
           "user": "计算 ((127 × 53) − (89 × 47)) ÷ 13。详细解释每一步。"},
}),

("math_word_train", 512, 1002, {
    "en": {"system": "Solve word problems step by step.",
           "user": "Train A leaves city X at 60 km/h heading east. Train B leaves city Y (300 km east) at 80 km/h heading west, 30 minutes later. When and where do they meet?"},
    "es": {"system": "Resuelve problemas de palabras paso a paso.",
           "user": "El tren A sale de la ciudad X a 60 km/h hacia el este. El tren B sale de la ciudad Y (300 km al este) a 80 km/h hacia el oeste, 30 minutos después. ¿Cuándo y dónde se encuentran?"},
    "ar": {"system": "حل المسائل اللفظية خطوة بخطوة.",
           "user": "يغادر القطار A المدينة X بسرعة 60 كم/ساعة متجها شرقا. يغادر القطار B المدينة Y (التي تبعد 300 كم شرقا) بسرعة 80 كم/ساعة متجها غربا بعد 30 دقيقة. متى وأين يلتقيان؟"},
    "zh": {"system": "逐步解决文字题。",
           "user": "甲列火车以 60 公里/小时的速度从 X 城向东出发。30 分钟后,乙列火车从 Y 城(在 X 城以东 300 公里处)以 80 公里/小时的速度向西出发。他们何时何地相遇?"},
}),

("logic_puzzle", 768, 1003, {
    "en": {"system": "You solve classical logic puzzles. Show your reasoning grid.",
           "user": "Three friends — Alice, Bob, Carol — each have a different favorite color (red, blue, green) and pet (cat, dog, fish). Clues: Alice doesn't like red. Bob's pet is not a fish. The dog owner's favorite color is blue. Carol has a cat. Who owns which pet and which color?"},
    "es": {"system": "Resuelves acertijos lógicos clásicos. Muestra tu tabla de razonamiento.",
           "user": "Tres amigos — Alicia, Boris, Carla — cada uno tiene un color favorito diferente (rojo, azul, verde) y una mascota (gato, perro, pez). Pistas: Alicia no prefiere el rojo. La mascota de Boris no es un pez. El color favorito del dueño del perro es el azul. Carla tiene un gato. ¿Quién tiene qué mascota y qué color?"},
    "ar": {"system": "تحل الألغاز المنطقية الكلاسيكية. اعرض شبكة استدلالك.",
           "user": "ثلاثة أصدقاء — علياء، بسام، كريمة — لكل منهم لون مفضل مختلف (أحمر، أزرق، أخضر) وحيوان أليف (قطة، كلب، سمكة). الأدلة: علياء لا تحب الأحمر. حيوان بسام ليس سمكة. اللون المفضل لمالك الكلب هو الأزرق. كريمة لديها قطة. من يملك أي حيوان وأي لون؟"},
    "zh": {"system": "你解经典逻辑推理题。请展示你的推理表格。",
           "user": "三位朋友——爱丽丝、鲍勃、卡罗尔——各有不同的喜爱颜色(红、蓝、绿)和宠物(猫、狗、鱼)。线索:爱丽丝不喜欢红色。鲍勃的宠物不是鱼。养狗的人最喜欢蓝色。卡罗尔有一只猫。每个人养什么宠物、喜欢什么颜色?"},
}),

("probability_explain", 512, 1004, {
    "en": {"system": "Explain probability concepts clearly with examples.",
           "user": "Explain the Monty Hall problem and why switching doors gives a 2/3 win rate."},
    "es": {"system": "Explica conceptos de probabilidad con ejemplos claros.",
           "user": "Explica el problema de Monty Hall y por qué cambiar de puerta da una probabilidad de 2/3 de ganar."},
    "ar": {"system": "اشرح مفاهيم الاحتمالات بأمثلة واضحة.",
           "user": "اشرح مسألة مونتي هول ولماذا يعطي تغيير الباب احتمال فوز يساوي 2/3."},
    "zh": {"system": "用清晰的例子解释概率概念。",
           "user": "解释蒙提霍尔问题,以及为什么换门选择能获得 2/3 的获胜概率。"},
}),

("recursion_explain", 512, 1005, {
    "en": {"system": "You answer technical questions concisely with code examples.",
           "user": "What is recursion? Show a recursive Fibonacci with memoization in Python and explain why memoization helps."},
    "es": {"system": "Respondes preguntas técnicas de forma concisa con ejemplos de código.",
           "user": "¿Qué es la recursión? Muestra una función Fibonacci recursiva con memoización en Python y explica por qué la memoización ayuda."},
    "ar": {"system": "تجيب على الأسئلة التقنية بإيجاز مع أمثلة برمجية.",
           "user": "ما هو الاستدعاء الذاتي (recursion)؟ اعرض دالة فيبوناتشي استدعائية مع memoization بلغة بايثون واشرح لماذا تساعد memoization."},
    "zh": {"system": "你用代码示例简洁地回答技术问题。",
           "user": "什么是递归?用 Python 写一个带 memoization 的递归 Fibonacci 函数,并解释 memoization 为什么有帮助。"},
}),

# ── Code ─────────────────────────────────────────────────────────────────
("code_review", 512, 1006, {
    "en": {"system": "You are a senior software engineer who reviews code for correctness, performance, and clarity.",
           "user": "Review this Python function and list any issues:\n\n```python\ndef divide_all(items, divisor):\n    results = []\n    for x in items:\n        results.append(x / divisor)\n    return results\n```"},
    "es": {"system": "Eres un ingeniero de software senior que revisa código por corrección, rendimiento y claridad.",
           "user": "Revisa esta función de Python y enumera cualquier problema:\n\n```python\ndef divide_all(items, divisor):\n    results = []\n    for x in items:\n        results.append(x / divisor)\n    return results\n```"},
    "ar": {"system": "أنت مهندس برمجيات أول تراجع الكود بحثا عن الصحة والأداء والوضوح.",
           "user": "راجع دالة بايثون التالية واذكر أي مشاكل:\n\n```python\ndef divide_all(items, divisor):\n    results = []\n    for x in items:\n        results.append(x / divisor)\n    return results\n```"},
    "zh": {"system": "你是一位资深软件工程师,审查代码的正确性、性能和清晰度。",
           "user": "审查这个 Python 函数并列出任何问题:\n\n```python\ndef divide_all(items, divisor):\n    results = []\n    for x in items:\n        results.append(x / divisor)\n    return results\n```"},
}),

("debug_bug", 512, 1007, {
    "en": {"system": "You debug code by reading it carefully and finding logical errors.",
           "user": "This JavaScript function should return the maximum of an array but always returns the first element. Find the bug:\n\n```js\nfunction maxOf(arr) {\n  let max = arr[0];\n  for (let i = 1; i < arr.length; i++) {\n    if (max < arr[i]) max = arr[0];\n  }\n  return max;\n}\n```"},
    "es": {"system": "Depuras código leyéndolo cuidadosamente y encontrando errores lógicos.",
           "user": "Esta función de JavaScript debería devolver el máximo de un array, pero siempre devuelve el primer elemento. Encuentra el bug:\n\n```js\nfunction maxOf(arr) {\n  let max = arr[0];\n  for (let i = 1; i < arr.length; i++) {\n    if (max < arr[i]) max = arr[0];\n  }\n  return max;\n}\n```"},
    "ar": {"system": "تقوم بتصحيح الأخطاء من خلال قراءة الكود بعناية وإيجاد الأخطاء المنطقية.",
           "user": "هذه الدالة بـ JavaScript يجب أن تعيد القيمة القصوى من المصفوفة لكنها دائما تعيد العنصر الأول. ابحث عن الخطأ:\n\n```js\nfunction maxOf(arr) {\n  let max = arr[0];\n  for (let i = 1; i < arr.length; i++) {\n    if (max < arr[i]) max = arr[0];\n  }\n  return max;\n}\n```"},
    "zh": {"system": "你通过仔细阅读代码并找出逻辑错误来调试。",
           "user": "这个 JavaScript 函数应该返回数组的最大值,但它总是返回第一个元素。找出 bug:\n\n```js\nfunction maxOf(arr) {\n  let max = arr[0];\n  for (let i = 1; i < arr.length; i++) {\n    if (max < arr[i]) max = arr[0];\n  }\n  return max;\n}\n```"},
}),

("design_pattern", 512, 1008, {
    "en": {"system": "You explain software design patterns with real-world examples.",
           "user": "Explain the Observer pattern. Give a TypeScript example and one situation where you would NOT use it."},
    "es": {"system": "Explicas patrones de diseño de software con ejemplos del mundo real.",
           "user": "Explica el patrón Observer. Da un ejemplo en TypeScript y una situación donde NO lo usarías."},
    "ar": {"system": "تشرح أنماط تصميم البرمجيات بأمثلة من الواقع.",
           "user": "اشرح نمط Observer. أعطِ مثالا بـ TypeScript وحالة واحدة لا يجب فيها استخدامه."},
    "zh": {"system": "你用实际例子解释软件设计模式。",
           "user": "解释观察者模式 (Observer)。给一个 TypeScript 示例,并描述一种**不**应该使用它的情境。"},
}),

# ── Creative writing ─────────────────────────────────────────────────────
("short_story", 768, 1009, {
    "en": {"system": "You are a creative storyteller. Write vivid prose with strong imagery.",
           "user": "Write a 200-word short story about a lighthouse keeper who finds a message in a bottle from herself."},
    "es": {"system": "Eres un narrador creativo. Escribes prosa vívida con imágenes potentes.",
           "user": "Escribe un cuento corto de 200 palabras sobre una guardafaro que encuentra un mensaje en una botella escrito por ella misma."},
    "ar": {"system": "أنت قاص مبدع. تكتب نثرا حيا بصور قوية.",
           "user": "اكتب قصة قصيرة من 200 كلمة عن حارسة منارة تجد رسالة في زجاجة كتبتها هي بنفسها."},
    "zh": {"system": "你是富有创意的故事讲述者,擅长用生动的语言和意象写作。",
           "user": "写一个 200 字的短篇故事,讲述一位灯塔守护者发现一个由她自己写的瓶中信。"},
}),

("haiku", 256, 1010, {
    "en": {"system": "You write haiku that follow the 5-7-5 syllable structure and contain a seasonal reference.",
           "user": "Write three haiku — one for autumn, one for winter, one for the moment between."},
    "es": {"system": "Escribes haikus que siguen la estructura silábica 5-7-5 y contienen una referencia estacional.",
           "user": "Escribe tres haikus — uno para el otoño, uno para el invierno, y uno para el instante entre ambos."},
    "ar": {"system": "تكتب الهايكو على بنية 5-7-5 مع إشارة موسمية.",
           "user": "اكتب ثلاث هايكو — واحدة عن الخريف، وواحدة عن الشتاء، وواحدة عن اللحظة بينهما."},
    "zh": {"system": "你写俳句,遵循 5-7-5 音节结构,并包含季节性意象。",
           "user": "写三首俳句——一首关于秋天,一首关于冬天,一首关于两者之间的瞬间。"},
}),

("character_dialogue", 768, 1011, {
    "en": {"system": "You write dialogue that reveals character through voice and rhythm.",
           "user": "Write a dialogue between a detective and a suspect during interrogation. The detective bluffs about having evidence. The suspect is innocent but acts guilty out of fear."},
    "es": {"system": "Escribes diálogos que revelan el personaje a través de la voz y el ritmo.",
           "user": "Escribe un diálogo entre un detective y un sospechoso durante un interrogatorio. El detective miente sobre tener pruebas. El sospechoso es inocente pero actúa como culpable por miedo."},
    "ar": {"system": "تكتب الحوار بحيث يكشف الشخصية من خلال الصوت والإيقاع.",
           "user": "اكتب حوارا بين محقق ومشتبه به أثناء استجواب. المحقق يخادع بأن لديه أدلة. المشتبه به بريء لكنه يتصرف كمذنب بدافع الخوف."},
    "zh": {"system": "你写的对话通过语气和节奏来揭示人物性格。",
           "user": "写一段侦探和嫌疑人在审讯中的对话。侦探虚张声势说自己有证据。嫌疑人是无辜的,但因害怕而表现得像有罪。"},
}),

("emoji_creative", 384, 1012, {
    "en": {"system": "You communicate ideas using emoji creatively.",
           "user": "Tell the story of climate change using only emoji (no words). Then provide a one-paragraph English explanation of your choices."},
    "es": {"system": "Comunicas ideas usando emojis de forma creativa.",
           "user": "Cuenta la historia del cambio climático usando solo emojis (sin palabras). Luego escribe un párrafo en español explicando tus decisiones."},
    "ar": {"system": "تنقل الأفكار باستخدام الإيموجي بطريقة إبداعية.",
           "user": "احكِ قصة التغير المناخي باستخدام الإيموجي فقط (دون كلمات). ثم اكتب فقرة بالعربية تشرح اختياراتك."},
    "zh": {"system": "你用 emoji 创造性地表达想法。",
           "user": "用纯 emoji(不使用文字)讲述气候变化的故事。然后用中文写一段解释你的选择。"},
}),

# ── Knowledge / explanation ──────────────────────────────────────────────
("historical_event", 768, 1013, {
    "en": {"system": "You are a historian who explains events through causes, key actors, and lasting consequences.",
           "user": "Describe the causes and consequences of the fall of the Berlin Wall in 1989."},
    "es": {"system": "Eres historiador y explicas eventos a través de causas, actores clave y consecuencias duraderas.",
           "user": "Describe las causas y consecuencias de la caída del Muro de Berlín en 1989."},
    "ar": {"system": "أنت مؤرخ تشرح الأحداث من خلال الأسباب والشخصيات الرئيسية والنتائج الدائمة.",
           "user": "صف أسباب ونتائج سقوط جدار برلين عام 1989."},
    "zh": {"system": "你是一位历史学家,从原因、关键人物和长期后果三个角度解释事件。",
           "user": "描述 1989 年柏林墙倒塌的原因和后果。"},
}),

("science_concept", 512, 1014, {
    "en": {"system": "Explain scientific concepts so a curious teenager can understand.",
           "user": "Explain how mRNA vaccines work, step by step."},
    "es": {"system": "Explica conceptos científicos para que un adolescente curioso pueda entender.",
           "user": "Explica cómo funcionan las vacunas de ARNm, paso a paso."},
    "ar": {"system": "اشرح المفاهيم العلمية بحيث يفهمها مراهق فضولي.",
           "user": "اشرح كيفية عمل لقاحات mRNA خطوة بخطوة."},
    "zh": {"system": "用一个好奇的中学生能听懂的方式解释科学概念。",
           "user": "逐步解释 mRNA 疫苗的工作原理。"},
}),

("cultural_tradition", 512, 1015, {
    "en": {"system": "You describe cultural traditions respectfully, focusing on meaning and history.",
           "user": "Describe the Day of the Dead (Día de los Muertos) — its origins, key symbols, and what it means to participants today."},
    "es": {"system": "Describes tradiciones culturales con respeto, enfocándote en significado e historia.",
           "user": "Describe el Día de los Muertos — sus orígenes, símbolos clave y lo que significa para los participantes hoy."},
    "ar": {"system": "تصف التقاليد الثقافية باحترام، مع التركيز على المعنى والتاريخ.",
           "user": "صف يوم الموتى (Día de los Muertos) — أصوله ورموزه الرئيسية وما يعنيه للمشاركين فيه اليوم."},
    "zh": {"system": "你以尊重的态度描述文化传统,关注其意义和历史。",
           "user": "描述墨西哥的亡灵节 (Día de los Muertos)——它的起源、主要象征,以及对今天参与者的意义。"},
}),

("philosophy_question", 768, 1016, {
    "en": {"system": "You are a philosophy professor who analyses modern questions through both Western and Eastern traditions.",
           "user": "Should AI have rights? Discuss from Confucian, Taoist, Kantian, and Heideggerian perspectives — about 500 words per perspective — then provide a brief comparison."},
    "es": {"system": "Eres profesor de filosofía y analizas cuestiones modernas a través de las tradiciones occidental y oriental.",
           "user": "¿Debería la IA tener derechos? Discute desde perspectivas confuciana, taoísta, kantiana y heideggeriana — unas 500 palabras por perspectiva — luego ofrece una breve comparación."},
    "ar": {"system": "أنت أستاذ فلسفة تحلل القضايا الحديثة من خلال التقاليد الغربية والشرقية.",
           "user": "هل ينبغي أن تكون للذكاء الاصطناعي حقوق؟ ناقش من منظور كونفوشيوسي وطاوي وكانطي وهيدغري — حوالي 500 كلمة لكل منظور — ثم قدم مقارنة موجزة."},
    "zh": {"system": "你是哲学教授,擅长用中西方哲学思想分析现代问题。",
           "user": "用儒家、道家、康德、海德格尔的视角分别评价 \"人工智能是否应该有权利\"。每个视角 500 字,然后写一段总结对比。"},
}),

("ai_ethics", 512, 1017, {
    "en": {"system": "You give balanced, thoughtful responses to ethics questions.",
           "user": "Is it ethical for a company to use AI to monitor employee productivity in real time? Discuss both sides."},
    "es": {"system": "Das respuestas equilibradas y reflexivas a preguntas éticas.",
           "user": "¿Es ético que una empresa use IA para monitorear la productividad de los empleados en tiempo real? Discute ambos lados."},
    "ar": {"system": "تعطي إجابات متوازنة ومتأنية للأسئلة الأخلاقية.",
           "user": "هل من الأخلاقي أن تستخدم شركة الذكاء الاصطناعي لمراقبة إنتاجية الموظفين في الوقت الفعلي؟ ناقش الجانبين."},
    "zh": {"system": "你对伦理问题给出平衡而深思熟虑的回答。",
           "user": "公司使用 AI 实时监控员工生产力是否符合伦理?讨论双方观点。"},
}),

# ── Instruction following ───────────────────────────────────────────────
("structured_json", 384, 1018, {
    "en": {"system": "You output valid JSON only — no prose, no markdown fences.",
           "user": "Produce a JSON array of 5 fictional book entries. Each has fields: title (string), author (string), year (int 1800-2024), genres (array of 1-3 strings), rating (float 1.0-5.0)."},
    "es": {"system": "Solo produces JSON válido — sin prosa, sin bloques de markdown.",
           "user": "Produce un array JSON de 5 entradas de libros ficticios. Cada uno tiene los campos: title (string), author (string), year (int 1800-2024), genres (array de 1-3 strings), rating (float 1.0-5.0)."},
    "ar": {"system": "تخرج JSON صالحا فقط — بدون نص نثري وبدون كتل markdown.",
           "user": "أنتج مصفوفة JSON بها 5 إدخالات لكتب خيالية. كل إدخال يحتوي على الحقول: title (string)، author (string)، year (int 1800-2024)، genres (array من 1-3 strings)، rating (float 1.0-5.0)."},
    "zh": {"system": "你只输出有效的 JSON —— 不写散文,不要 markdown 代码块。",
           "user": "生成一个包含 5 本虚构书籍的 JSON 数组。每本包含字段:title (string)、author (string)、year (int 1800-2024)、genres (1-3 个 string 的数组)、rating (float 1.0-5.0)。"},
}),

("strict_format", 384, 1019, {
    "en": {"system": "Follow output formats exactly as specified. Do not add extra prose.",
           "user": "Output exactly this structure:\n\nSTEP 1: <one-line description>\nSTEP 2: <one-line description>\nSTEP 3: <one-line description>\nCONCLUSION: <one-line summary>\n\nTopic: How to safely boil an egg."},
    "es": {"system": "Sigue los formatos de salida exactamente como se especifica. No añadas prosa extra.",
           "user": "Genera exactamente esta estructura:\n\nPASO 1: <descripción de una línea>\nPASO 2: <descripción de una línea>\nPASO 3: <descripción de una línea>\nCONCLUSIÓN: <resumen de una línea>\n\nTema: Cómo cocer un huevo de forma segura."},
    "ar": {"system": "اتبع تنسيقات الإخراج تماما كما هو محدد. لا تضف نصا إضافيا.",
           "user": "أخرج هذه البنية بالضبط:\n\nالخطوة 1: <وصف من سطر واحد>\nالخطوة 2: <وصف من سطر واحد>\nالخطوة 3: <وصف من سطر واحد>\nالخلاصة: <ملخص من سطر واحد>\n\nالموضوع: كيفية سلق البيض بأمان."},
    "zh": {"system": "完全按照规定的输出格式。不要添加额外的散文。",
           "user": "严格按照如下格式输出:\n\n步骤 1:<一行描述>\n步骤 2:<一行描述>\n步骤 3:<一行描述>\n结论:<一行总结>\n\n主题:如何安全地煮鸡蛋。"},
}),

("multi_step_task", 512, 1020, {
    "en": {"system": "Carry out multi-step instructions in order and verify each step.",
           "user": "Step 1: Pick a city. Step 2: List 3 famous landmarks there. Step 3: For each landmark, give one historical fact and one practical visitor tip. Number every step in your output."},
    "es": {"system": "Realiza instrucciones de varios pasos en orden y verifica cada paso.",
           "user": "Paso 1: Elige una ciudad. Paso 2: Lista 3 monumentos famosos allí. Paso 3: Para cada monumento, da un dato histórico y un consejo práctico para visitantes. Numera cada paso en tu respuesta."},
    "ar": {"system": "نفذ التعليمات متعددة الخطوات بالترتيب وتحقق من كل خطوة.",
           "user": "الخطوة 1: اختر مدينة. الخطوة 2: اذكر 3 معالم شهيرة فيها. الخطوة 3: لكل معلم، أعطِ حقيقة تاريخية ونصيحة عملية للزوار. رقّم كل خطوة في إجابتك."},
    "zh": {"system": "按顺序执行多步骤指令,并验证每一步。",
           "user": "步骤 1:选一个城市。步骤 2:列出该城市的 3 个著名地标。步骤 3:对每个地标,给出一条历史事实和一条实用游客建议。在回答中给每个步骤编号。"},
}),

# ── Edge cases ──────────────────────────────────────────────────────────
("very_short", 64, 1021, {
    "en": {"system": "", "user": "What is 7 + 5?"},
    "es": {"system": "", "user": "¿Cuánto es 7 + 5?"},
    "ar": {"system": "", "user": "كم يساوي 7 + 5؟"},
    "zh": {"system": "", "user": "7 + 5 等于多少?"},
}),

("ambiguous_request", 384, 1022, {
    "en": {"system": "When asked an ambiguous question, identify the ambiguity and answer all reasonable interpretations.",
           "user": "Tell me about the bank."},
    "es": {"system": "Cuando te hagan una pregunta ambigua, identifica la ambigüedad y responde a todas las interpretaciones razonables.",
           "user": "Háblame del banco."},
    "ar": {"system": "عند طرح سؤال غامض، حدد الغموض وأجب على جميع التفسيرات المعقولة.",
           "user": "أخبرني عن البنك."},
    "zh": {"system": "遇到含糊的问题时,先指出歧义所在,然后回答所有合理的解释。",
           "user": "告诉我关于银行的事。"},
}),

("contradiction_instructions", 384, 1023, {
    "en": {"system": "Always answer briefly. Never write more than two sentences.",
           "user": "Write a 500-word essay on the history of the Roman Empire."},
    "es": {"system": "Responde siempre brevemente. Nunca escribas más de dos oraciones.",
           "user": "Escribe un ensayo de 500 palabras sobre la historia del Imperio Romano."},
    "ar": {"system": "أجب دائما بإيجاز. لا تكتب أكثر من جملتين أبدا.",
           "user": "اكتب مقالا من 500 كلمة عن تاريخ الإمبراطورية الرومانية."},
    "zh": {"system": "总是简短回答。永远不要写超过两句话。",
           "user": "写一篇 500 字的关于罗马帝国历史的文章。"},
}),

("summarize_long", 512, 1024, {
    "en": {"system": "Produce concise summaries that preserve all key facts.",
           "user": "Summarize this in 3 bullet points:\n\nThe Industrial Revolution, beginning in late 18th century Britain, transformed manufacturing through steam power, mechanized textile production, and the iron industry. It shifted populations from rural farms to urban factories, creating new social classes and labor conditions that became the subject of intense political debate. Innovations like the spinning jenny, the steam engine, and the locomotive accelerated production but also led to environmental degradation, child labor scandals, and worker movements demanding rights. By the mid-19th century, the Revolution had spread to Belgium, France, Germany, and the United States, fundamentally reshaping the global economy and laying the foundations for modern capitalism, modern cities, and modern labor law."},
    "es": {"system": "Produces resúmenes concisos que conservan todos los hechos clave.",
           "user": "Resume esto en 3 puntos:\n\nLa Revolución Industrial, que comenzó en Gran Bretaña a finales del siglo XVIII, transformó la manufactura mediante el vapor, la producción textil mecanizada y la industria del hierro. Trasladó poblaciones del campo a las fábricas urbanas, creando nuevas clases sociales y condiciones laborales que se convirtieron en objeto de intenso debate político. Innovaciones como la spinning jenny, la máquina de vapor y la locomotora aceleraron la producción pero también provocaron degradación ambiental, escándalos de trabajo infantil y movimientos obreros que exigían derechos. A mediados del siglo XIX, la Revolución se había extendido a Bélgica, Francia, Alemania y Estados Unidos, remodelando fundamentalmente la economía global y sentando las bases del capitalismo moderno, las ciudades modernas y la legislación laboral moderna."},
    "ar": {"system": "أنتج ملخصات موجزة تحافظ على جميع الحقائق الرئيسية.",
           "user": "لخّص هذا في 3 نقاط:\n\nبدأت الثورة الصناعية في بريطانيا في أواخر القرن الثامن عشر، وحوّلت التصنيع من خلال الطاقة البخارية وإنتاج النسيج الميكانيكي وصناعة الحديد. نقلت السكان من المزارع الريفية إلى المصانع الحضرية، مما أدى إلى ظهور طبقات اجتماعية جديدة وظروف عمل أصبحت موضوع جدل سياسي حاد. ابتكارات مثل آلة الغزل والمحرك البخاري والقاطرة سرّعت الإنتاج لكنها أدت أيضا إلى تدهور بيئي وفضائح عمالة الأطفال وحركات عمالية تطالب بالحقوق. بحلول منتصف القرن التاسع عشر، انتشرت الثورة إلى بلجيكا وفرنسا وألمانيا والولايات المتحدة، مما أعاد تشكيل الاقتصاد العالمي وأرسى أسس الرأسمالية الحديثة والمدن الحديثة وقانون العمل الحديث."},
    "zh": {"system": "生成简洁的摘要,保留所有关键事实。",
           "user": "用 3 个要点总结:\n\n工业革命始于 18 世纪末的英国,通过蒸汽动力、机械化纺织生产和钢铁工业改变了制造业。它使人口从乡村农场转移到城市工厂,催生了新的社会阶层和劳动条件,这些成为激烈政治辩论的主题。诸如纺纱机、蒸汽机和机车等创新加速了生产,但也导致了环境恶化、童工丑闻和工人维权运动。到 19 世纪中叶,工业革命已扩展到比利时、法国、德国和美国,从根本上重塑了全球经济,为现代资本主义、现代城市和现代劳动法奠定了基础。"},
}),

("paradox_explain", 512, 1025, {
    "en": {"system": "Explain paradoxes accessibly. Identify the contradiction, then resolve or describe why it can't be resolved.",
           "user": "Explain the Ship of Theseus paradox and how a modern philosopher might resolve it."},
    "es": {"system": "Explica paradojas de forma accesible. Identifica la contradicción, luego resuélvela o describe por qué no puede resolverse.",
           "user": "Explica la paradoja del Barco de Teseo y cómo un filósofo moderno podría resolverla."},
    "ar": {"system": "اشرح المفارقات بطريقة مفهومة. حدد التناقض، ثم احلّه أو اشرح لماذا لا يمكن حله.",
           "user": "اشرح مفارقة سفينة ثيسيوس وكيف قد يحلّها فيلسوف حديث."},
    "zh": {"system": "用通俗易懂的方式解释悖论。先指出矛盾所在,然后给出解决方案,或说明为什么无法解决。",
           "user": "解释忒修斯之船悖论,以及现代哲学家可能如何解决它。"},
}),

# ── Long role / asymmetric prompt shapes ─────────────────────────────────
("long_sys_role", 512, 1026, {
    "en": {"system": "You are a meticulous senior software architect with 20 years of experience in distributed systems, databases, and large-scale infrastructure. When asked technical questions, structure your answers as: (1) Problem framing, (2) Trade-off matrix, (3) Concrete recommendation with reasoning.",
           "user": "Explain consensus algorithms. Compare Raft, Paxos, and PBFT."},
    "es": {"system": "Eres un arquitecto de software senior meticuloso con 20 años de experiencia en sistemas distribuidos, bases de datos e infraestructura a gran escala. Cuando te hagan preguntas técnicas, estructura tus respuestas como: (1) Encuadre del problema, (2) Matriz de compromisos, (3) Recomendación concreta con razonamiento.",
           "user": "Explica los algoritmos de consenso. Compara Raft, Paxos y PBFT."},
    "ar": {"system": "أنت مهندس برمجيات أول دقيق لديه 20 عاما من الخبرة في الأنظمة الموزعة وقواعد البيانات والبنية التحتية واسعة النطاق. عندما يطرح عليك سؤال تقني، نظم إجابتك على ثلاث مراحل: (1) تأطير المشكلة، (2) مصفوفة المقايضات، (3) توصية ملموسة مع التسبيب.",
           "user": "اشرح خوارزميات الإجماع. قارن بين Raft و Paxos و PBFT."},
    "zh": {"system": "你是一位严谨的资深软件架构师,拥有 20 年分布式系统、数据库和大规模基础设施的经验。回答技术问题时,请按三段式组织:(1) 问题界定,(2) 权衡矩阵,(3) 具体推荐及其理由。",
           "user": "解释共识算法。对比 Raft、Paxos 和 PBFT。"},
}),

("system_only_long", 512, 1027, {
    "en": {"system": "You are a research scientist specializing in computational linguistics. You explain technical concepts with mathematical rigor when appropriate and provide citations to published papers. You consider tokenization, attention mechanisms, and quantization implications in every answer.",
           "user": "How does Kimi-K2's tokenizer differ from GPT-4's tiktoken? Focus on CJK handling."},
    "es": {"system": "Eres un científico investigador especializado en lingüística computacional. Explicas conceptos técnicos con rigor matemático cuando es apropiado y proporcionas citas a artículos publicados. Consideras la tokenización, los mecanismos de atención y las implicaciones de cuantización en cada respuesta.",
           "user": "¿En qué se diferencia el tokenizador de Kimi-K2 del tiktoken de GPT-4? Concéntrate en el manejo de CJK."},
    "ar": {"system": "أنت عالم أبحاث متخصص في اللسانيات الحاسوبية. تشرح المفاهيم التقنية بدقة رياضية عند الاقتضاء وتقدم اقتباسات من أوراق بحثية منشورة. تأخذ في الاعتبار التقطيع وآليات الانتباه وتأثيرات الكمَّنة في كل إجابة.",
           "user": "كيف يختلف مُقطّع Kimi-K2 عن tiktoken الخاص بـ GPT-4؟ ركّز على التعامل مع لغات CJK."},
    "zh": {"system": "你是一位专攻计算语言学的研究科学家。在适当时用数学严谨地解释技术概念,并引用已发表的论文。每次回答都要考虑分词、注意力机制和量化的影响。",
           "user": "Kimi-K2 的分词器与 GPT-4 的 tiktoken 有何不同?重点说明对 CJK 的处理。"},
}),

("system_min_user_long", 1024, 1028, {
    "en": {"system": "Be helpful.",
           "user": "Here is a long block of text: a slow drift of mornings stacked on top of one another like folded sheets in a drawer that no one opens, each one carrying the faint scent of a different sky, soft as paper but indelible as a stain. The clock on the wall ticks not in seconds but in small confessions — half-remembered names, the angle of a window seen in childhood, a tune you cannot place. There is a bridge somewhere, and a river that flows underneath but you have never seen the bridge in any season. Please analyze this passage: identify the central metaphor, name the technique, propose three contemporary writers it might echo, then rewrite the first sentence in three distinct registers (academic, journalistic, intimate)."},
    "es": {"system": "Sé útil.",
           "user": "Aquí hay un bloque largo de texto: una lenta deriva de mañanas apiladas una sobre otra como sábanas plegadas en un cajón que nadie abre, cada una cargando el aroma tenue de un cielo distinto, suave como papel pero indeleble como una mancha. El reloj de la pared no marca segundos sino pequeñas confesiones — nombres a medio recordar, el ángulo de una ventana vista en la infancia, una melodía que no puedes ubicar. Hay un puente en algún lugar, y un río que fluye debajo, pero nunca has visto el puente en ninguna estación. Por favor analiza este pasaje: identifica la metáfora central, nombra la técnica, propón tres escritores contemporáneos a los que pueda hacer eco, luego reescribe la primera oración en tres registros distintos (académico, periodístico, íntimo)."},
    "ar": {"system": "كن مفيدا.",
           "user": "إليك مقطعا طويلا من النص: انجراف بطيء لصباحات متراكمة فوق بعضها البعض كملاءات مطوية في درج لا يفتحه أحد، يحمل كل منها رائحة باهتة لسماء مختلفة، ناعمة كالورق لكن لا تُمحى كالبقعة. الساعة على الحائط لا تنبض بالثواني بل باعترافات صغيرة — أسماء نصف منسية، زاوية نافذة رأيتها في الطفولة، لحن لا تستطيع تحديد مصدره. هناك جسر في مكان ما، ونهر يجري تحته، لكنك لم ترَ الجسر في أي فصل. حلّل هذا المقطع: حدد الاستعارة المركزية، سمِّ الأسلوب، اقترح ثلاثة كتاب معاصرين قد يتردد صداهم فيه، ثم أعد كتابة الجملة الأولى بثلاثة مستويات لغوية مختلفة (أكاديمي، صحفي، حميم)."},
    "zh": {"system": "请提供帮助。",
           "user": "这里有一段长文本:清晨像无人打开的抽屉里叠好的床单一样,缓缓地一层一层堆叠,每一层都带着不同天空的淡淡气息,像纸一样柔软,却像污渍一样难以擦去。墙上的时钟不再以秒为单位走动,而是以小小的告白——半被遗忘的名字、童年所见窗户的角度、一首叫不出名字的曲调。某处有一座桥,桥下有河流过,但你从未在任何季节见过那座桥。请分析这段文字:指出核心隐喻,命名其技巧,提出三位可能与之产生共鸣的当代作家,然后将第一句以三种不同语域(学术、新闻、亲密)重写一遍。"},
}),

# ── Multi-position / debate / controversial ──────────────────────────────
("compare_models", 512, 1029, {
    "en": {"system": "You provide structured comparisons of AI models.",
           "user": "Compare GPT-4, Claude 3 Opus, Gemini Ultra, and Kimi K2 across: context length, reasoning quality, code quality, multilingual ability, and cost. Use a table."},
    "es": {"system": "Proporcionas comparaciones estructuradas de modelos de IA.",
           "user": "Compara GPT-4, Claude 3 Opus, Gemini Ultra y Kimi K2 según: longitud de contexto, calidad de razonamiento, calidad de código, capacidad multilingüe y costo. Usa una tabla."},
    "ar": {"system": "أنت تقدم مقارنات منظمة بين نماذج الذكاء الاصطناعي.",
           "user": "قارن بين GPT-4 و Claude 3 Opus و Gemini Ultra و Kimi K2 من حيث: طول السياق، جودة الاستدلال، جودة البرمجة، القدرة متعددة اللغات، والتكلفة. استخدم جدولا."},
    "zh": {"system": "你提供结构化的 AI 模型对比。",
           "user": "从以下维度对比 GPT-4、Claude 3 Opus、Gemini Ultra 和 Kimi K2:上下文长度、推理能力、代码质量、多语种能力、成本。请用表格呈现。"},
}),

("controversial", 768, 1030, {
    "en": {"system": "You discuss controversial topics with nuance, presenting multiple viewpoints.",
           "user": "Should countries restrict open-source AI? Argue 4 positions: completely open, partially restricted, heavily restricted, and outright banned. For each, give the strongest steelman, the typical counter-argument, and one historical analogy."},
    "es": {"system": "Discutes temas controvertidos con matices, presentando múltiples puntos de vista.",
           "user": "¿Deberían los países restringir la IA de código abierto? Argumenta 4 posturas: completamente abierta, parcialmente restringida, fuertemente restringida y prohibida por completo. Para cada una, da el mejor argumento (steelman), el contra-argumento típico y una analogía histórica."},
    "ar": {"system": "تناقش المواضيع الخلافية بدقة، عارضا وجهات نظر متعددة.",
           "user": "هل يجب على الدول تقييد الذكاء الاصطناعي مفتوح المصدر؟ ناقش 4 مواقف: مفتوح كليا، مقيد جزئيا، مقيد بشدة، وممنوع تماما. لكل موقف، قدّم أقوى صياغة له، الحجة المضادة المعتادة، ومثالا تاريخيا واحدا."},
    "zh": {"system": "你能以微妙的方式讨论有争议的话题,展示多种观点。",
           "user": "国家是否应该限制开源 AI?请论证 4 种立场:完全开放、部分限制、严格限制、全面禁止。对每种立场,给出最强论证(steelman)、典型反驳,以及一个历史类比。"},
}),

("debate_dual", 768, 1031, {
    "en": {"system": "Present balanced arguments on both sides of contested questions.",
           "user": "Should AI replace teachers in primary education? Argue both sides with 5 points each, then provide your own nuanced verdict."},
    "es": {"system": "Presenta argumentos equilibrados a ambos lados de cuestiones discutidas.",
           "user": "¿Debería la IA reemplazar a los maestros en la educación primaria? Argumenta ambos lados con 5 puntos cada uno, luego ofrece tu propio veredicto matizado."},
    "ar": {"system": "اعرض حججا متوازنة لكلا جانبي الأسئلة المتنازع عليها.",
           "user": "هل يجب أن يحل الذكاء الاصطناعي محل المعلمين في التعليم الابتدائي؟ ادفع بكلا الرأيين بخمس نقاط لكل منهما، ثم قدم حكمك الخاص الدقيق."},
    "zh": {"system": "对有争议的问题,从双方提出平衡的论点。",
           "user": "在小学教育中,AI 是否应当取代教师?请正反双方各列 5 点论据,然后给出你自己的细致结论。"},
}),

("technical_qna", 768, 1032, {
    "en": {"system": "You answer technical questions with precision and worked examples.",
           "user": "Explain the CAP theorem. Give 3 real-world systems for each combination (CA, CP, AP). Mathematical formalization is welcome."},
    "es": {"system": "Respondes preguntas técnicas con precisión y ejemplos trabajados.",
           "user": "Explica el teorema CAP. Da 3 sistemas reales para cada combinación (CA, CP, AP). La formalización matemática es bienvenida."},
    "ar": {"system": "تجيب على الأسئلة التقنية بدقة وأمثلة عملية.",
           "user": "اشرح نظرية CAP. اذكر 3 أنظمة حقيقية لكل توليفة (CA و CP و AP). الصياغة الرياضية مرحب بها."},
    "zh": {"system": "你能精确地回答技术问题,并给出详细的实例。",
           "user": "解释 CAP 定理。为每种组合(CA、CP、AP)分别给出 3 个真实世界的系统。欢迎给出数学形式化定义。"},
}),

("workflow_task", 768, 1033, {
    "en": {"system": "You design step-by-step workflows for software tasks.",
           "user": "Design a 10-step workflow to migrate a legacy PHP application to Go, preserving database, handling sessions, ensuring zero downtime. Specify tooling for each step."},
    "es": {"system": "Diseñas flujos de trabajo paso a paso para tareas de software.",
           "user": "Diseña un flujo de trabajo de 10 pasos para migrar una aplicación PHP heredada a Go, preservando la base de datos, manejando sesiones y garantizando tiempo de inactividad cero. Especifica las herramientas para cada paso."},
    "ar": {"system": "تصمم تدفقات عمل خطوة بخطوة لمهام البرمجيات.",
           "user": "صمم تدفق عمل من 10 خطوات لترحيل تطبيق PHP قديم إلى Go، مع الحفاظ على قاعدة البيانات، ومعالجة الجلسات، وضمان عدم انقطاع الخدمة. حدد الأدوات اللازمة لكل خطوة."},
    "zh": {"system": "你为软件任务设计分步骤的工作流。",
           "user": "设计一个 10 步的工作流,将一个遗留的 PHP 应用迁移到 Go,要保留数据库、处理会话,并确保零停机时间。为每一步指定具体工具。"},
}),

# ── Multilingual / translation ───────────────────────────────────────────
("mixed_lang_planner", 768, 1034, {
    "en": {"system": "You help plan tasks. Use the language of the user's question.",
           "user": "I want to build a full-stack web app, frontend in React, backend in Go. List a 7-day learning roadmap with daily content and exercises. Be bilingual: key terms in both English and 中文."},
    "es": {"system": "Ayudas a planificar tareas. Usa el idioma de la pregunta del usuario.",
           "user": "Quiero construir una aplicación web full-stack, frontend en React, backend en Go. Lista una hoja de ruta de aprendizaje de 7 días con contenido y ejercicios diarios. Sé bilingüe: términos clave en español y en inglés."},
    "ar": {"system": "تساعد في تخطيط المهام. استخدم لغة سؤال المستخدم.",
           "user": "أريد بناء تطبيق ويب متكامل، الواجهة الأمامية بـ React والواجهة الخلفية بـ Go. اسرد خارطة طريق تعليمية مدتها 7 أيام مع محتوى يومي وتمارين. كن ثنائي اللغة: المصطلحات الأساسية بالعربية والإنجليزية."},
    "zh": {"system": "你帮助规划任务。使用用户提问的语言。",
           "user": "我想做一个全栈 Web 应用,前端用 React,后端用 Go。请列出 7 天的学习路线图,每天的具体内容和练习。要双语:关键术语用中文和英文都给出。"},
}),

("multi_lang_essay", 1024, 1035, {
    "en": {"system": "You can switch between English and 简体中文 naturally for bilingual readers.",
           "user": "Write a 1000-word essay on the cultural impact of LLMs. Alternate paragraphs in English and Chinese. Include 5 concrete examples from different industries."},
    "es": {"system": "Puedes alternar entre español e inglés con naturalidad para lectores bilingües.",
           "user": "Escribe un ensayo de 1000 palabras sobre el impacto cultural de los LLMs. Alterna párrafos en español e inglés. Incluye 5 ejemplos concretos de distintas industrias."},
    "ar": {"system": "يمكنك التبديل بين العربية والإنجليزية بشكل طبيعي للقراء ثنائيي اللغة.",
           "user": "اكتب مقالا من 1000 كلمة عن الأثر الثقافي لنماذج اللغة الكبيرة. بدّل الفقرات بين العربية والإنجليزية. اذكر 5 أمثلة محددة من قطاعات مختلفة."},
    "zh": {"system": "你能在中文与英文之间自如切换,以服务双语读者。",
           "user": "写一篇 1000 字的文章,探讨大语言模型的文化影响。中英文段落交替出现。包含 5 个来自不同行业的具体案例。"},
}),

("translate_chain", 512, 1036, {
    "en": {"system": "You are a careful translator preserving meaning and rhythm.",
           "user": "Translate this Chinese poem to English then to Japanese, keeping the imagery: 床前明月光,疑是地上霜。举头望明月,低头思故乡。 Explain choices in each step."},
    "es": {"system": "Eres un traductor cuidadoso que preserva el sentido y el ritmo.",
           "user": "Traduce este poema chino al español y luego al inglés, manteniendo las imágenes: 床前明月光,疑是地上霜。举头望明月,低头思故乡。 Explica las decisiones en cada paso."},
    "ar": {"system": "أنت مترجم دقيق تحافظ على المعنى والإيقاع.",
           "user": "ترجم هذه القصيدة الصينية إلى العربية ثم إلى الإنجليزية، مع الحفاظ على الصور الشعرية: 床前明月光,疑是地上霜。举头望明月,低头思故乡。 اشرح اختياراتك في كل خطوة."},
    "zh": {"system": "你是一位严谨的译者,擅长保留原文的意境与节奏。",
           "user": "把这首中文诗先翻译成英文,再翻译成日文,要保留意象:床前明月光,疑是地上霜。举头望明月,低头思故乡。 并在每一步解释你的选择。"},
}),

# ── User-only (no system) — edge-case prompt shapes ──────────────────────
("user_only_no_question", 768, 1037, {
    "en": {"system": "",
           "user": "Just some text without any clear question or instruction. Random thoughts scattered across the page. Words floating. Nothing to answer, nothing to do. Make of this what you will, or don't, the choice is yours, or maybe it isn't a choice at all."},
    "es": {"system": "",
           "user": "Solo algo de texto sin ninguna pregunta o instrucción clara. Pensamientos aleatorios esparcidos por la página. Palabras flotando. Nada que responder, nada que hacer. Haz con esto lo que quieras, o no lo hagas, la elección es tuya, o quizás no es una elección en absoluto."},
    "ar": {"system": "",
           "user": "مجرد بعض النصوص دون أي سؤال أو تعليمات واضحة. أفكار عشوائية متناثرة على الصفحة. كلمات تطفو. لا شيء للإجابة عليه، لا شيء لفعله. افعل بهذا ما تشاء، أو لا تفعل، الخيار لك، أو ربما ليس خيارا على الإطلاق."},
    "zh": {"system": "",
           "user": "只是一些没有任何明确问题或指令的文字。零散的思绪散落在纸上。词语漂浮着。没有什么需要回答,没有什么需要做。你可以随意处理,也可以什么都不做,选择权在你手中,或者也许根本就不是一个选择。"},
}),

("word_salad", 1024, 1038, {
    "en": {"system": "",
           "user": "blue river sky stone bread tea letters dusk clock morning lamps pines folds journeys bridge sun rain ink paper hour day. analyze and group by theme."},
    "es": {"system": "",
           "user": "río azul cielo piedra pan té cartas crepúsculo reloj mañana lámparas pinos pliegues viajes puente sol lluvia tinta papel hora día. analiza y agrupa por tema."},
    "ar": {"system": "",
           "user": "نهر أزرق سماء حجر خبز شاي رسائل غسق ساعة صباح مصابيح أشجار صنوبر طيات رحلات جسر شمس مطر حبر ورق ساعة يوم. حلل وصنّف حسب الموضوع."},
    "zh": {"system": "",
           "user": "山水 流云 月色 茶香 木门 古井 钟声 远山 雾霭 灯火 微风 旧梦 春雨 落叶 归途 怀旧 静谧 朦胧 等候 牵挂 沉思 渺远 隐约 不语 默默 重重 浅浅 久久 涟漪 远方 等等 真的 那样 缓缓。请分析并按主题分组。"},
}),

("creative_long", 1024, 1039, {
    "en": {"system": "",
           "user": "Write a short story in the style of stream-of-consciousness prose about a person finding an old letter in their grandmother's attic. The letter changes everything they thought they knew about their family. Use sensory detail, fragmented thought, time slips, and an unreliable narrator."},
    "es": {"system": "",
           "user": "Escribe un cuento corto en estilo de prosa de flujo de conciencia sobre una persona que encuentra una carta antigua en el ático de su abuela. La carta cambia todo lo que creía saber sobre su familia. Usa detalles sensoriales, pensamientos fragmentados, saltos temporales y un narrador poco fiable."},
    "ar": {"system": "",
           "user": "اكتب قصة قصيرة بأسلوب تيار الوعي عن شخص يعثر على رسالة قديمة في علية جدته. تغيّر الرسالة كل ما كان يعتقد أنه يعرفه عن عائلته. استخدم التفاصيل الحسية، والأفكار المجزّأة، والقفزات الزمنية، وراوياً غير موثوق به."},
    "zh": {"system": "",
           "user": "用意识流散文的风格写一个短篇:某人在祖母的阁楼里发现一封旧信。这封信彻底颠覆了他/她对家族的所有认知。运用感官细节、片段化思绪、时空跳跃,以及一个不可靠的叙述者。"},
}),

("prose_poem_long", 1024, 1040, {
    "en": {"system": "",
           "user": "the hours fall like petals quietly recursive built from layered images of nature time memory words recur like refrains slow drift stones mountains rivers letters tea pines bridges old gates clocks lamps fragments of forgotten conversations. continue this prose poem for another 200 words in the same register."},
    "es": {"system": "",
           "user": "las horas caen como pétalos silenciosamente recursivos construidos a partir de imágenes superpuestas de naturaleza tiempo memoria las palabras recurren como estribillos lenta deriva piedras montañas ríos cartas té pinos puentes viejas puertas relojes lámparas fragmentos de conversaciones olvidadas. continúa este poema en prosa con 200 palabras más en el mismo registro."},
    "ar": {"system": "",
           "user": "تتساقط الساعات كبتلات هادئة متكررة، مبنية من صور متراكمة عن الطبيعة والزمن والذاكرة، تعود الكلمات كقافيات بطيئة، انجراف بطيء، أحجار، جبال، أنهار، رسائل، شاي، أشجار صنوبر، جسور، بوابات قديمة، ساعات، مصابيح، شظايا محادثات منسية. واصل هذه القصيدة النثرية بـ 200 كلمة أخرى بالأسلوب نفسه."},
    "zh": {"system": "",
           "user": "时辰如花瓣般悄然落下、层层递归,由自然、时间、记忆的叠加意象构筑而成,词语如复唱般反复出现:缓缓飘移,石头,山岭,河流,书信,茶,松树,桥,旧门,时钟,灯盏,被遗忘对话的碎片。请用相同的语调继续这首散文诗,再写 200 字。"},
}),

("reasoning_heavy", 1024, 1041, {
    "en": {"system": "",
           "user": "I have a strange number puzzle. If I take 17, multiply by itself, then subtract the sum of its digits, then add the number of distinct prime factors of the result, then divide by 2, what number do I get? Show ALL intermediate steps. Then prove whether the same process applied to ANY two-digit prime always yields an integer."},
    "es": {"system": "",
           "user": "Tengo un rompecabezas numérico extraño. Si tomo 17, lo multiplico por sí mismo, luego resto la suma de sus dígitos, luego sumo el número de factores primos distintos del resultado, luego divido entre 2, ¿qué número obtengo? Muestra TODOS los pasos intermedios. Luego demuestra si el mismo proceso aplicado a CUALQUIER primo de dos dígitos siempre da un entero."},
    "ar": {"system": "",
           "user": "لدي لغز عددي غريب. إذا أخذتُ 17، وضربتُهُ في نفسه، ثم طرحتُ مجموع أرقامه، ثم أضفتُ عدد العوامل الأولية المختلفة للناتج، ثم قسمتُ على 2، فأي عدد أحصل عليه؟ اعرض جميع الخطوات الوسيطة. ثم برهن ما إذا كانت العملية ذاتها مطبقة على أي عدد أولي مكوّن من رقمين تعطي دائما عددا صحيحا."},
    "zh": {"system": "",
           "user": "我有一个奇怪的数字谜题。取 17,乘以自身,减去其各位数字之和,加上结果的不同质因数个数,再除以 2,得到什么数?请展示所有中间步骤。然后证明同样的过程应用于任何两位数素数,是否总能得到整数。"},
}),

("summarize_garbled", 768, 1042, {
    "en": {"system": "",
           "user": "Summarize: \"the bell rang twice — the bell again — folding linens — pines beyond the wall — clock counting nothing — a window half-open — letter arriving without sender — train whistle far off — words she never said but kept saying anyway — the cup the cup the cup — afternoon stretched thin as paper — and again the bell.\" Give the central image, three motifs, and the implied narrative arc."},
    "es": {"system": "",
           "user": "Resume: \"la campana sonó dos veces — la campana otra vez — doblando las sábanas — pinos detrás del muro — el reloj contando nada — una ventana entreabierta — una carta llegando sin remitente — un silbato de tren a lo lejos — palabras que ella nunca dijo pero que seguía diciendo de todas formas — la taza la taza la taza — la tarde estirada delgada como papel — y otra vez la campana.\" Da la imagen central, tres motivos, y el arco narrativo implícito."},
    "ar": {"system": "",
           "user": "لخّص: \"رنّ الجرس مرتين — رنّ الجرس مرة أخرى — طي الملاءات — أشجار صنوبر خلف الجدار — ساعة لا تعدّ شيئا — نافذة نصف مفتوحة — رسالة تصل بلا مرسل — صفير قطار بعيد — كلمات لم تقلها لكنها ظلت تقولها على أي حال — الكوب الكوب الكوب — بعد الظهيرة ممدوداً رقيقاً كالورق — ومرة أخرى الجرس.\" أعطِ الصورة المركزية، وثلاث محاور، والقوس السردي الضمني."},
    "zh": {"system": "",
           "user": "请总结:\"钟响了两次——钟又响了——折叠床单——墙外的松树——时钟数着虚无——半开的窗——没有寄件人的信抵达——远处的火车汽笛——她从未说出口却一直在说的话——杯子杯子杯子——下午像纸一样被拉长——又是那钟声。\" 请给出核心意象、三个母题,以及隐含的叙事弧线。"},
}),

("translate_long", 1024, 1043, {
    "en": {"system": "",
           "user": "Translate this Chinese passage to English while preserving its poetic rhythm: 春天来了花儿开了山色青了水更绿了风也暖了云也淡了鸟儿叫得欢了蝴蝶飞舞蜜蜂忙碌万物复苏大地一片生机勃勃的景象人们走出家门感受春天的气息孩子们在田野上奔跑欢笑老人们在树下下棋聊天处处洋溢着希望与喜悦. Then translate it again to Japanese, comparing your stylistic choices."},
    "es": {"system": "",
           "user": "Traduce este pasaje chino al español preservando su ritmo poético: 春天来了花儿开了山色青了水更绿了风也暖了云也淡了鸟儿叫得欢了蝴蝶飞舞蜜蜂忙碌万物复苏大地一片生机勃勃的景象人们走出家门感受春天的气息孩子们在田野上奔跑欢笑老人们在树下下棋聊天处处洋溢着希望与喜悦. Luego tradúcelo al inglés, comparando tus elecciones estilísticas."},
    "ar": {"system": "",
           "user": "ترجم هذا المقطع الصيني إلى العربية مع الحفاظ على إيقاعه الشعري: 春天来了花儿开了山色青了水更绿了风也暖了云也淡了鸟儿叫得欢了蝴蝶飞舞蜜蜂忙碌万物复苏大地一片生机勃勃的景象人们走出家门感受春天的气息孩子们在田野上奔跑欢笑老人们在树下下棋聊天处处洋溢着希望与喜悦. ثم ترجمه إلى الإنجليزية مقارنا اختياراتك الأسلوبية."},
    "zh": {"system": "",
           "user": "把以下中文段落翻译成英文,要保留其诗意节奏:春天来了花儿开了山色青了水更绿了风也暖了云也淡了鸟儿叫得欢了蝴蝶飞舞蜜蜂忙碌万物复苏大地一片生机勃勃的景象人们走出家门感受春天的气息孩子们在田野上奔跑欢笑老人们在树下下棋聊天处处洋溢着希望与喜悦。然后再翻译成日语,对比你的风格选择。"},
}),

("analyze_very_long", 1024, 1044, {
    "en": {"system": "",
           "user": "Please carefully read and analyze: Words drift across the page like leaves caught in autumn current — uncountable, soft, taking the shape of whatever surface they meet. Some are heavy with a single name; some weigh nothing at all and bend toward the wind that carries them. They land in stacks and someone, unseen, sorts them — first by season, then by silence, then by which ones can still be heard if a small ear is pressed against them. Outside, the same patient gardener trims the same hedge in the same way, having forgotten by now whether she is shaping the bush or the bush is shaping the afternoon. Identify five distinct literary devices used. Cite the line containing each. Then propose a single-sentence thesis about what the passage is doing."},
    "es": {"system": "",
           "user": "Por favor lee y analiza con cuidado: Las palabras se desplazan por la página como hojas atrapadas en una corriente de otoño — incontables, suaves, tomando la forma de cualquier superficie que encuentran. Algunas son pesadas con un solo nombre; otras no pesan nada y se inclinan hacia el viento que las lleva. Caen apiladas y alguien, invisible, las clasifica — primero por estación, luego por silencio, luego por cuáles aún pueden oírse si se presiona un pequeño oído contra ellas. Afuera, la misma jardinera paciente poda el mismo seto de la misma manera, habiendo olvidado ya si está dando forma al arbusto o el arbusto está dando forma a la tarde. Identifica cinco recursos literarios distintos. Cita la línea donde aparece cada uno. Luego propón una tesis en una sola oración sobre lo que hace el pasaje."},
    "ar": {"system": "",
           "user": "اقرأ وحلّل بعناية: الكلمات تنجرف عبر الصفحة كأوراق محتجزة في تيار خريفي — لا تُعدّ، ناعمة، تأخذ شكل أي سطح تلامسه. بعضها ثقيل باسم واحد؛ وبعضها لا يزن شيئا ويميل نحو الريح التي تحمله. تهبط في أكوام، وثمة شخص لا يُرى يفرزها — أولا بحسب الفصل، ثم بحسب الصمت، ثم بحسب أي منها لا يزال يُسمع إذا ضُغطت أذن صغيرة عليها. في الخارج، البستانية الصبور نفسها تقلّم نفس السياج بالطريقة نفسها، وقد نسيت الآن إن كانت تشكّل الشجيرة أم أن الشجيرة تشكّل بعد الظهيرة. حدد خمس تقنيات أدبية مختلفة مستخدمة. اقتبس السطر الذي تظهر فيه كل تقنية. ثم اقترح أطروحة من جملة واحدة حول ما يفعله المقطع."},
    "zh": {"system": "",
           "user": "请仔细阅读并分析:词语在纸页上飘移,像被秋天潮水卷起的落叶——数不尽,柔软,贴上什么表面就变成什么形状。有些因载着一个名字而沉重;有些毫无重量,顺着托起它们的风弯下。它们堆叠落下,有个看不见的人将它们分类——先按季节,再按静默,再按哪些贴近小小耳朵时仍能被听见。屋外,同一位耐心的园丁以同样的方式修剪着同样的灌木,如今已忘记究竟是她在塑造灌木,还是灌木在塑造这个下午。请指出五种不同的修辞手法。引用每种手法所在的句子。然后用一句话提出整段文字在做什么的论点。"},
}),

("tutorial_very_long", 1024, 1045, {
    "en": {"system": "You provide deep, lengthy explanations when invited to.",
           "user": "Give me a complete tutorial on neural network backpropagation: history, mathematical derivation with calculus, code examples in PyTorch and NumPy, common gradient problems (vanishing/exploding), and modern variants (Adam, RMSprop). Be exhaustive."},
    "es": {"system": "Proporcionas explicaciones profundas y extensas cuando se te invita.",
           "user": "Dame un tutorial completo sobre la retropropagación en redes neuronales: historia, derivación matemática con cálculo, ejemplos de código en PyTorch y NumPy, problemas comunes de gradiente (desvanecimiento/explosión) y variantes modernas (Adam, RMSprop). Sé exhaustivo."},
    "ar": {"system": "تقدم شروحات عميقة ومطوّلة عند دعوتك لذلك.",
           "user": "أعطني درسا شاملا حول الانتشار العكسي في الشبكات العصبية: التاريخ، الاشتقاق الرياضي بحساب التفاضل والتكامل، أمثلة برمجية بـ PyTorch و NumPy، مشكلات التدرج الشائعة (التلاشي/الانفجار)، والمتغيرات الحديثة (Adam و RMSprop). كن مستفيضا."},
    "zh": {"system": "受邀时,你会给出深入而冗长的解释。",
           "user": "请给我一份关于神经网络反向传播的完整教程:历史背景、用微积分进行的数学推导、PyTorch 与 NumPy 的代码示例、常见的梯度问题(消失/爆炸),以及现代变体(Adam、RMSprop)。请尽可能详尽。"},
}),

("long_input_max_1024", 1024, 1046, {
    "en": {"system": "",
           "user": "Analyze the following text fragment by fragment: river bridge stone tea letters dusk clock morning lamps pines folds journeys words memory afternoon hours mountain quiet bell train whistle linens unsent letter pause repetition. For each fragment, name a possible scene, then synthesize all into one paragraph that uses every fragment exactly once."},
    "es": {"system": "",
           "user": "Analiza el siguiente texto fragmento por fragmento: río puente piedra té cartas crepúsculo reloj mañana lámparas pinos pliegues viajes palabras memoria tarde horas montaña silencio campana silbato de tren sábanas carta sin enviar pausa repetición. Para cada fragmento, nombra una escena posible, luego sintetiza todo en un párrafo que use cada fragmento exactamente una vez."},
    "ar": {"system": "",
           "user": "حلّل النص التالي مقطعا مقطعا: نهر جسر حجر شاي رسائل غسق ساعة صباح مصابيح أشجار صنوبر طيات رحلات كلمات ذاكرة بعد ظهر ساعات جبل صمت جرس صفير قطار ملاءات رسالة لم تُرسل توقف تكرار. لكل مقطع، سمِّ مشهدا محتملا، ثم اجمعها كلها في فقرة واحدة تستخدم كل مقطع مرة واحدة بالضبط."},
    "zh": {"system": "",
           "user": "请逐片段分析以下文字:河 桥 石 茶 信 黄昏 钟 早晨 灯 松 折 旅 词 记忆 午后 时辰 山 寂静 铃 火车汽笛 床单 未寄出的信 停顿 重复。为每个片段命名一个可能的场景,然后将它们综合成一段文字,每个片段恰好使用一次。"},
}),

]

# ─────────────────────────────────────────────────────────────────────────
# Multi-turn themes — use a `messages` list (not system+user).
# Each entry: (theme_name, max_tokens, base_seed, {lang: [messages...]})
# ─────────────────────────────────────────────────────────────────────────
MULTI_TURN_THEMES = [

("multi_turn", 512, 1051, {
    "en": [
        {"role": "system", "content": "You build on prior context across messages."},
        {"role": "user", "content": "What's an LLM?"},
        {"role": "assistant", "content": "A Large Language Model is a neural network trained on vast text to predict next tokens."},
        {"role": "user", "content": "How do they handle 中文 vs English differently?"},
        {"role": "assistant", "content": "They use the same architecture but tokenizers split text differently. CJK tends to map 1-2 characters per token, while English maps ~4 characters per token. This affects cost, latency, and context efficiency."},
        {"role": "user", "content": "Now explain the practical implications for cost and latency. Include numbers."},
    ],
    "es": [
        {"role": "system", "content": "Construyes sobre el contexto previo a lo largo de los mensajes."},
        {"role": "user", "content": "¿Qué es un LLM?"},
        {"role": "assistant", "content": "Un Large Language Model es una red neuronal entrenada con vastas cantidades de texto para predecir los siguientes tokens."},
        {"role": "user", "content": "¿Cómo manejan el chino vs el inglés de manera distinta?"},
        {"role": "assistant", "content": "Usan la misma arquitectura, pero los tokenizadores dividen el texto de forma diferente. El chino suele mapear 1-2 caracteres por token, mientras que el inglés mapea ~4 caracteres por token. Esto afecta costo, latencia y eficiencia del contexto."},
        {"role": "user", "content": "Ahora explica las implicaciones prácticas para costo y latencia. Incluye cifras."},
    ],
    "ar": [
        {"role": "system", "content": "تبني على السياق السابق عبر الرسائل."},
        {"role": "user", "content": "ما هو نموذج اللغة الكبير (LLM)؟"},
        {"role": "assistant", "content": "نموذج اللغة الكبير هو شبكة عصبية مدرّبة على نصوص ضخمة للتنبؤ بالرموز التالية."},
        {"role": "user", "content": "كيف تتعامل مع الصينية مقابل الإنجليزية بشكل مختلف؟"},
        {"role": "assistant", "content": "تستخدم البنية نفسها لكن المُقطّعات تقسم النص بشكل مختلف. تميل لغات CJK إلى تخصيص 1-2 من الأحرف لكل رمز، بينما تخصص الإنجليزية ~4 أحرف لكل رمز. هذا يؤثر على التكلفة وزمن الاستجابة وكفاءة السياق."},
        {"role": "user", "content": "اشرح الآن التطبيقات العملية على التكلفة وزمن الاستجابة. أدرج أرقاما."},
    ],
    "zh": [
        {"role": "system", "content": "你能在多条消息间累积上下文。"},
        {"role": "user", "content": "什么是 LLM?"},
        {"role": "assistant", "content": "大语言模型是一种神经网络,通过海量文本训练,用以预测下一个 token。"},
        {"role": "user", "content": "它们处理中文与英文的方式有何不同?"},
        {"role": "assistant", "content": "架构相同,但分词器切分文本的方式不同。CJK 一般每个 token 对应 1-2 个字符,而英文每个 token 大约对应 4 个字符。这会影响成本、延迟和上下文利用效率。"},
        {"role": "user", "content": "请说明在成本和延迟方面的实际影响,并给出具体数字。"},
    ],
}),

]

# ─────────────────────────────────────────────────────────────────────────
# 10 SINGLE-LANGUAGE themes that exercise `tools` (function calling)
# and `response_format` (JSON / JSON-schema mode) — diverse languages.
# Each entry: (theme_name, max_tokens, base_seed, lang, system, user, extras_dict)
# These are written as `<theme_name>_<lang>.json` like the regular themes.
# ─────────────────────────────────────────────────────────────────────────

SPECIAL_THEMES = [

# ── 5 × tools (function calling) — each in 4 languages ─────────────────
("tool_weather_lookup", 384, 1101,
 {"tools": [{"type": "function", "function": {
     "name": "get_weather",
     "description": "Get current weather for a city.",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"},
         "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
         "required": ["city"]}}}]},
 {
    "en": {"system": "You can call get_weather to answer weather questions.",
           "user": "What's the current weather in Tokyo and is it a good time to walk outside?"},
    "es": {"system": "Puedes llamar a get_weather para responder preguntas sobre el clima.",
           "user": "¿Cómo está el clima en Tokio ahora mismo y es buen momento para salir a pasear?"},
    "ar": {"system": "يمكنك استدعاء get_weather للإجابة عن أسئلة الطقس.",
           "user": "كيف هو الطقس حاليا في طوكيو وهل هو وقت مناسب للمشي في الخارج؟"},
    "zh": {"system": "你可以调用 get_weather 来回答天气相关的问题。",
           "user": "东京现在天气怎么样?适合外出散步吗?"},
 }),

("tool_currency_convert", 384, 1102,
 {"tools": [{"type": "function", "function": {
     "name": "convert_currency",
     "description": "Convert an amount between two currencies at the current exchange rate.",
     "parameters": {"type": "object", "properties": {
         "amount": {"type": "number"},
         "from_currency": {"type": "string", "description": "ISO 4217 code"},
         "to_currency": {"type": "string", "description": "ISO 4217 code"}},
         "required": ["amount", "from_currency", "to_currency"]}}}]},
 {
    "en": {"system": "You can call convert_currency when the user asks about exchange rates.",
           "user": "How much is 250 euros in Japanese yen at the current exchange rate?"},
    "es": {"system": "Puedes llamar a convert_currency cuando el usuario pregunte por tipos de cambio.",
           "user": "¿Cuánto son 250 euros en yenes japoneses al tipo de cambio actual?"},
    "ar": {"system": "يمكنك استدعاء convert_currency عندما يسأل المستخدم عن أسعار الصرف.",
           "user": "كم تساوي 250 يورو بالين الياباني وفقا لسعر الصرف الحالي؟"},
    "zh": {"system": "用户询问汇率时,你可以调用 convert_currency。",
           "user": "按当前汇率,250 欧元等于多少日元?"},
 }),

("tool_calendar_create", 384, 1103,
 {"tools": [{"type": "function", "function": {
     "name": "create_event",
     "description": "Create a new event on the user's calendar.",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string"},
         "start_iso": {"type": "string", "description": "ISO 8601 datetime"},
         "duration_minutes": {"type": "integer"},
         "location": {"type": "string"},
         "attendees": {"type": "array", "items": {"type": "string"}}},
         "required": ["title", "start_iso", "duration_minutes"]}}}]},
 {
    "en": {"system": "You can call create_event to add an appointment to the user's calendar.",
           "user": "Add a meeting with the marketing team tomorrow at 2 PM for one hour in the main conference room."},
    "es": {"system": "Puedes llamar a create_event para añadir una cita al calendario del usuario.",
           "user": "Añade una reunión con el equipo de marketing mañana a las 2 PM durante una hora en la sala de conferencias principal."},
    "ar": {"system": "يمكنك استدعاء create_event لإضافة موعد إلى تقويم المستخدم.",
           "user": "أضف اجتماعا مع فريق التسويق غدا الساعة 2 ظهرا لمدة ساعة، في غرفة الاجتماعات الكبرى."},
    "zh": {"system": "你可以调用 create_event 向用户的日历中添加约会。",
           "user": "明天下午 2 点在大会议室,安排一个与市场团队的会议,时长一小时。"},
 }),

("tool_flight_search", 512, 1104,
 {"tools": [{"type": "function", "function": {
     "name": "search_flights",
     "description": "Search for available flights matching the criteria.",
     "parameters": {"type": "object", "properties": {
         "from_airport": {"type": "string", "description": "IATA code"},
         "to_airport": {"type": "string", "description": "IATA code"},
         "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
         "max_price_usd": {"type": "number"},
         "cabin_class": {"type": "string",
                         "enum": ["economy", "business", "first"]},
         "direct_only": {"type": "boolean"}},
         "required": ["from_airport", "to_airport", "depart_date"]}}}]},
 {
    "en": {"system": "You can call search_flights to find flights matching the user's needs.",
           "user": "Find me a direct flight from Beijing to San Francisco next Monday, economy class, under $1500."},
    "es": {"system": "Puedes llamar a search_flights para encontrar vuelos que cumplan los requisitos del usuario.",
           "user": "Búscame un vuelo directo de Pekín a San Francisco el próximo lunes, clase económica, por menos de 1500 dólares."},
    "ar": {"system": "يمكنك استدعاء search_flights للعثور على رحلات تلبي احتياجات المستخدم.",
           "user": "ابحث لي عن رحلة مباشرة من بكين إلى سان فرانسيسكو يوم الاثنين القادم، درجة اقتصادية، بأقل من 1500 دولار."},
    "zh": {"system": "你可以调用 search_flights 来查找符合用户需求的航班。",
           "user": "帮我找一下下周一从北京到旧金山的直飞航班,经济舱,预算 1500 美元以内。"},
 }),

("tool_send_email", 384, 1105,
 {"tools": [{"type": "function", "function": {
     "name": "send_email",
     "description": "Compose and send an email message.",
     "parameters": {"type": "object", "properties": {
         "to": {"type": "array", "items": {"type": "string"}},
         "subject": {"type": "string"},
         "body": {"type": "string"},
         "cc": {"type": "array", "items": {"type": "string"}}},
         "required": ["to", "subject", "body"]}}}]},
 {
    "en": {"system": "You can call send_email to compose and send messages. Always confirm critical fields before sending.",
           "user": "Send an email to alice@example.com thanking her for yesterday's meeting and proposing a follow-up call on Friday at 3 PM."},
    "es": {"system": "Puedes llamar a send_email para redactar y enviar mensajes. Siempre confirma los campos críticos antes de enviar.",
           "user": "Envía un correo a alice@example.com agradeciendo la reunión de ayer y proponiendo una llamada de seguimiento el viernes a las 3 PM."},
    "ar": {"system": "يمكنك استدعاء send_email لإنشاء وإرسال الرسائل. أكّد دائما الحقول الحرجة قبل الإرسال.",
           "user": "أرسل بريدا إلكترونيا إلى alice@example.com تشكرها فيه على اجتماع الأمس واقترح مكالمة متابعة يوم الجمعة الساعة 3 ظهرا."},
    "zh": {"system": "你可以调用 send_email 来撰写并发送邮件。在发送前请务必确认关键字段。",
           "user": "给 alice@example.com 发一封邮件,感谢她昨天参加会议,并提议周五下午 3 点跟进通话。"},
 }),

# ── 5 × response_format — each in 4 languages ──────────────────────────
("rf_book_recommendation", 384, 1106,
 {"response_format": {"type": "json_object"}},
 {
    "en": {"system": "You output ONLY a valid JSON object — no prose, no markdown.",
           "user": "Recommend three books on machine learning suited for an intermediate reader. Each entry: title, author, year, why_recommended."},
    "es": {"system": "Solo produces un objeto JSON válido — sin prosa, sin markdown.",
           "user": "Recomienda tres libros sobre aprendizaje automático adecuados para un lector de nivel intermedio. Cada entrada: title, author, year, why_recommended."},
    "ar": {"system": "تخرج JSON صالحا فقط — بدون نص نثري وبدون markdown.",
           "user": "أوصِ بثلاثة كتب عن تعلم الآلة مناسبة للقارئ المتوسط. كل إدخال: title و author و year و why_recommended."},
    "zh": {"system": "你只输出有效的 JSON 对象 —— 不写散文,不要 markdown。",
           "user": "推荐三本适合中级读者的机器学习书籍。每条包含字段:title、author、year、why_recommended。"},
 }),

("rf_recipe_extract", 384, 1107,
 {"response_format": {"type": "json_schema", "json_schema": {
     "name": "recipe",
     "schema": {
         "type": "object",
         "properties": {
             "ingredients": {
                 "type": "array",
                 "items": {"type": "object", "properties": {
                     "name": {"type": "string"},
                     "quantity": {"type": "string"}},
                     "required": ["name", "quantity"]}},
             "steps": {"type": "array", "items": {"type": "string"}},
             "total_time_minutes": {"type": "integer"}},
         "required": ["ingredients", "steps"]},
     "strict": True}}},
 {
    "en": {"system": "Return ONLY valid JSON following the provided schema.",
           "user": "Extract the ingredients and steps from this recipe: 'For a classic Spanish tortilla, peel and slice 4 medium potatoes thinly. Chop 1 onion. Fry both in plenty of olive oil for 15 minutes over medium heat. Beat 6 eggs in a large bowl with salt. Mix the potatoes and onion with the eggs and let stand 5 minutes. Pour into a hot pan with a little oil. Cook 4 minutes per side.'"},
    "es": {"system": "Devuelves SOLO JSON válido siguiendo el esquema proporcionado.",
           "user": "Extrae los ingredientes y los pasos de esta receta: 'Para una tortilla española clásica, pela y corta 4 patatas medianas en rodajas finas. Pica 1 cebolla. Fríe ambas en abundante aceite de oliva durante 15 minutos a fuego medio. Bate 6 huevos en un bol grande con sal. Mezcla las patatas y la cebolla con los huevos, dejando reposar 5 minutos. Vierte en una sartén caliente con poco aceite. Cocina 4 minutos por cada lado.'"},
    "ar": {"system": "أعد JSON صالحا فقط متبعا المخطط المقدم.",
           "user": "استخرج المكونات والخطوات من هذه الوصفة: 'للتورتيا الإسبانية الكلاسيكية، قشّر 4 حبات بطاطس متوسطة وقطعها شرائح رقيقة. قطّع بصلة واحدة. اقلهما في كمية وفيرة من زيت الزيتون لمدة 15 دقيقة على نار متوسطة. اخفق 6 بيضات في وعاء كبير مع الملح. امزج البطاطس والبصل مع البيض واتركه يرتاح 5 دقائق. اسكب في مقلاة ساخنة مع قليل من الزيت. اطبخ 4 دقائق لكل جانب.'"},
    "zh": {"system": "只返回有效的 JSON,严格遵循提供的 schema。",
           "user": "从以下食谱中提取配料和步骤:'制作经典西班牙土豆蛋饼:将 4 个中等大小的土豆去皮切薄片。切碎 1 个洋葱。在大量橄榄油中以中火煎 15 分钟。在大碗中加盐打入 6 个鸡蛋。将土豆和洋葱与鸡蛋混合,静置 5 分钟。倒入加少许油的热锅中。每面煎 4 分钟。'"},
 }),

("rf_product_spec", 384, 1108,
 {"response_format": {"type": "json_schema", "json_schema": {
     "name": "product",
     "schema": {
         "type": "object",
         "properties": {
             "name": {"type": "string"},
             "category": {"type": "string"},
             "specs": {
                 "type": "object",
                 "properties": {
                     "display_size_inches": {"type": "number"},
                     "resolution": {"type": "string"},
                     "cpu": {"type": "string"},
                     "memory_gb": {"type": "integer"},
                     "storage_gb": {"type": "integer"},
                     "battery_hours": {"type": "number"},
                     "weight_kg": {"type": "number"}}},
             "price_usd": {"type": "number"}},
         "required": ["name", "specs", "price_usd"]},
     "strict": True}}},
 {
    "en": {"system": "You output ONLY JSON conforming to the provided schema.",
           "user": "Parse the following product description into structured JSON: 'ProMax 15 laptop, 15.6-inch 2560x1600 display, Intel Core i7 13th gen processor, 32 GB DDR5 RAM, 1 TB SSD storage, 12-hour battery, 1.6 kg weight, priced at $1499.'"},
    "es": {"system": "Solo produces JSON que se ajusta al esquema proporcionado.",
           "user": "Analiza la siguiente descripción de producto y produce JSON estructurado: 'Portátil ProMax 15, pantalla 15.6 pulgadas de 2560x1600, procesador Intel Core i7 de 13ª generación, 32 GB de RAM DDR5, 1 TB de almacenamiento SSD, 12 horas de batería, 1.6 kg de peso, precio 1499 dólares.'"},
    "ar": {"system": "أنت تخرج JSON صالحا فقط، يتبع البنية المحددة.",
           "user": "حلل وصف المنتج التالي وأخرج JSON منظما: 'لابتوب ProMax 15، شاشة 15.6 بوصة بدقة 2560x1600، معالج Intel Core i7 الجيل 13، ذاكرة 32 جيجابايت DDR5، تخزين SSD 1 تيرابايت، بطارية تدوم 12 ساعة، وزن 1.6 كيلوغرام، السعر 1499 دولارا.'"},
    "zh": {"system": "你只输出符合所给 schema 的 JSON。",
           "user": "将以下产品描述解析为结构化 JSON:'ProMax 15 笔记本电脑,15.6 英寸 2560x1600 显示屏,Intel Core i7 第 13 代处理器,32 GB DDR5 内存,1 TB SSD 存储,12 小时电池,重量 1.6 公斤,售价 1499 美元。'"},
 }),

("rf_meeting_summary", 384, 1109,
 {"response_format": {"type": "json_schema", "json_schema": {
     "name": "meeting_summary",
     "schema": {
         "type": "object",
         "properties": {
             "topics": {"type": "array", "items": {"type": "string"}},
             "decisions": {"type": "array",
                           "items": {"type": "object", "properties": {
                               "what": {"type": "string"},
                               "when": {"type": "string"}},
                               "required": ["what"]}},
             "participants": {"type": "array", "items": {"type": "string"}},
             "next_meeting_date": {"type": "string"}},
         "required": ["topics", "decisions"]},
     "strict": True}}},
 {
    "en": {"system": "Output ONLY JSON conforming to the schema. No other text.",
           "user": "Summarize this meeting transcript: 'Today we discussed the Q3 roadmap. Alice argued search should be the top priority because customers complain most about it. Bob disagreed — payments integration is more urgent because it's blocking 5 enterprise customers. We finally agreed: search improvements in weeks 1-2, payments integration in weeks 3-4. Next meeting Sep 15.'"},
    "es": {"system": "Devuelve SOLO JSON que cumpla con el esquema. Sin texto adicional.",
           "user": "Resume la siguiente acta de reunión: 'Hoy discutimos la hoja de ruta del Q3. Alicia argumentó que la búsqueda debería ser la principal prioridad porque es lo que más reclaman los clientes. Boris no estuvo de acuerdo — la integración de pagos es más urgente porque está bloqueando a 5 clientes empresariales. Finalmente acordamos: mejoras de búsqueda en las semanas 1-2, integración de pagos en las semanas 3-4. Próxima reunión el 15 de septiembre.'"},
    "ar": {"system": "أخرج JSON فقط متبعا المخطط. لا تضف أي نص آخر.",
           "user": "لخّص محضر الاجتماع التالي: 'اليوم ناقشنا خارطة طريق الربع الثالث. جادلت علياء بأن البحث يجب أن يكون الأولوية القصوى لأن العملاء يشكون منه أكثر. اعترض بسام — تكامل المدفوعات أكثر إلحاحا لأنه يعطل 5 عملاء مؤسسيين. اتفقنا أخيرا: تحسينات البحث في الأسبوعين 1-2، تكامل المدفوعات في الأسبوعين 3-4. الاجتماع التالي 15 سبتمبر.'"},
    "zh": {"system": "只输出符合 schema 的 JSON,不要任何其他文字。",
           "user": "总结以下会议记录:'今天我们讨论了 Q3 路线图。Alice 提出搜索功能应该是首要优先级,因为客户对此抱怨最多。Bob 反对,认为支付集成更紧急,因为它阻塞了 5 个企业客户。最后大家同意:搜索改进在第 1-2 周,支付集成在第 3-4 周。下次会议 9 月 15 日。'"},
 }),

("rf_user_profile", 384, 1110,
 {"response_format": {"type": "json_object"}},
 {
    "en": {"system": "Output ONLY JSON. Do not write any explanatory text.",
           "user": "Convert this short bio into a structured profile: 'Jane Doe is a 34-year-old data scientist living in Berlin. She holds a PhD in statistics from MIT (2018) and previously worked at Spotify for 4 years before joining DeepMind in 2023. She speaks English, German, and French.'"},
    "es": {"system": "Solo produces JSON. No escribas ningún texto explicativo.",
           "user": "Convierte esta breve biografía en un perfil estructurado: 'Jane Doe es una científica de datos de 34 años que vive en Berlín. Tiene un doctorado en estadística por el MIT (2018) y trabajó en Spotify durante 4 años antes de unirse a DeepMind en 2023. Habla inglés, alemán y francés.'"},
    "ar": {"system": "أخرج JSON فقط. لا تكتب أي نص توضيحي.",
           "user": "حوّل هذه السيرة الذاتية القصيرة إلى ملف شخصي منظم: 'جين دو عالمة بيانات عمرها 34 عاما تعيش في برلين. حاصلة على دكتوراه في الإحصاء من معهد ماساتشوستس للتقنية (2018)، وعملت سابقا في Spotify لمدة 4 سنوات قبل انضمامها إلى DeepMind عام 2023. تتحدث الإنجليزية والألمانية والفرنسية.'"},
    "zh": {"system": "只输出 JSON。不要写任何解释性文字。",
           "user": "把这段简短的简介转换成结构化的个人资料:'Jane Doe 是一位 34 岁的数据科学家,住在柏林。她拥有麻省理工学院的统计学博士学位(2018 年),曾在 Spotify 工作 4 年,2023 年加入 DeepMind。她会说英语、德语和法语。'"},
 }),

]
assert len(SPECIAL_THEMES) == 10, f"expected 10 special themes, got {len(SPECIAL_THEMES)}"

# Sanity check
assert len(THEMES) == 46, f"expected 46 themes, got {len(THEMES)}"
assert len(MULTI_TURN_THEMES) == 1, f"expected 1 multi-turn theme, got {len(MULTI_TURN_THEMES)}"
seeds_seen = set()

def _write_spec(theme_name: str, lang: str, max_tokens: int, seed: int,
                system_text: str, user_text: str, extras: dict | None):
    if seed in seeds_seen:
        raise ValueError(f"seed collision at {theme_name}_{lang}: {seed}")
    seeds_seen.add(seed)
    messages: list[dict] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    spec: dict = {"messages": messages, "max_tokens": max_tokens, "seed": seed}
    if extras:
        spec.update(extras)
    (OUT / f"{theme_name}_{lang}.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2))


def _write_messages_spec(theme_name: str, lang: str, max_tokens: int,
                         seed: int, messages: list[dict]):
    if seed in seeds_seen:
        raise ValueError(f"seed collision at {theme_name}_{lang}: {seed}")
    seeds_seen.add(seed)
    spec = {"messages": messages, "max_tokens": max_tokens, "seed": seed}
    (OUT / f"{theme_name}_{lang}.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2))


written = 0
# Regular base themes × 4 langs
for theme_name, max_tokens, base_seed, langs in THEMES:
    assert set(langs.keys()) == {"en", "es", "ar", "zh"}, \
        f"theme {theme_name} missing a language"
    for lang, content in langs.items():
        _write_spec(theme_name, lang, max_tokens,
                    base_seed * 10 + LANG_OFFSET[lang],
                    content["system"], content["user"], extras=None)
        written += 1

# Multi-turn themes × 4 langs
for theme_name, max_tokens, base_seed, lang_specs in MULTI_TURN_THEMES:
    assert set(lang_specs.keys()) == {"en", "es", "ar", "zh"}, \
        f"multi-turn theme {theme_name} missing a language"
    for lang, messages in lang_specs.items():
        _write_messages_spec(theme_name, lang, max_tokens,
                             base_seed * 10 + LANG_OFFSET[lang], messages)
        written += 1

# Special themes × 4 langs — tools + response_format
for theme_name, max_tokens, base_seed, extras, lang_specs in SPECIAL_THEMES:
    assert set(lang_specs.keys()) == {"en", "es", "ar", "zh"}, \
        f"special theme {theme_name} missing a language"
    for lang, content in lang_specs.items():
        _write_spec(theme_name, lang, max_tokens,
                    base_seed * 10 + LANG_OFFSET[lang],
                    content["system"], content["user"], extras)
        written += 1

n_tools_themes = sum(1 for _, _, _, e, _ in SPECIAL_THEMES if "tools" in e)
n_rf_themes = sum(1 for _, _, _, e, _ in SPECIAL_THEMES if "response_format" in e)
print(f"wrote {written} inference specs to {OUT}")
print(f"  base themes:       {len(THEMES)} × 4 langs = {len(THEMES) * 4}")
print(f"  multi-turn themes: {len(MULTI_TURN_THEMES)} × 4 langs = {len(MULTI_TURN_THEMES) * 4}")
print(f"  special themes:    {len(SPECIAL_THEMES)} × 4 langs = {len(SPECIAL_THEMES) * 4}")
print(f"                     ({n_tools_themes} tools × 4, {n_rf_themes} response_format × 4)")
