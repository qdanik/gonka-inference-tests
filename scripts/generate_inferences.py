"""Generate 100 inference prompts: 25 themes × 4 languages (en/es/ar/zh).

Each theme is preserved semantically across languages. Outputs to
`inferences/<theme>_<lang>.json` with shape `{messages, max_tokens, seed}`.

Run from repo root: `python3 scripts/generate_inferences.py`
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "inferences"
OUT.mkdir(exist_ok=True)

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
assert len(THEMES) == 25, f"expected 25 themes, got {len(THEMES)}"
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


written = 0
# Regular 25 × 4 = 100
for theme_name, max_tokens, base_seed, langs in THEMES:
    assert set(langs.keys()) == {"en", "es", "ar", "zh"}, \
        f"theme {theme_name} missing a language"
    for lang, content in langs.items():
        _write_spec(theme_name, lang, max_tokens,
                    base_seed * 10 + LANG_OFFSET[lang],
                    content["system"], content["user"], extras=None)
        written += 1

# 10 special themes × 4 langs = 40 — tools + response_format
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
print(f"  regular themes: {len(THEMES)} × 4 langs = {len(THEMES) * 4}")
print(f"  special themes: {len(SPECIAL_THEMES)} × 4 langs = {len(SPECIAL_THEMES) * 4}")
print(f"    ({n_tools_themes} tools × 4, {n_rf_themes} response_format × 4)")
