// One plain-language explanation per idea, used by every tooltip, so a term
// means the same thing wherever it appears.

export const GLOSSARY = {
  token: "The pieces GPT-2 reads: a word, part of a word, or punctuation. ␣ marks a space at the start of a token.",
  start: "⟨start⟩ is GPT-2's <|endoftext|> token. Featurama puts it in front of every prompt, as researchers do. Heads with nothing to fetch often rest their attention on it.",
  next: "GPT-2's five likeliest next tokens and how likely it thinks each one is. Tab adds the first; click any of them to add it.",
  layer: "GPT-2 small runs 12 layers in order, numbered 0 to 11. Each one reads every token's running summary and adds to it.",
  head: "An attention head moves information from earlier tokens to later ones. Every layer has 12, each with its own idea of where to look.",
  mlp: "Each layer's MLP works on every token by itself, adding what it knows about that token in context. It never looks at other tokens.",
  residual: "Each token's running summary: 768 numbers that every layer reads and adds to. Drawn as a blue line.",
  lens: "The logit lens reads GPT-2's best guess part-way through, by running the model's final step early on the running summary.",
  activity: "How much the part adds to the running summary at the last token, measured as the length of what it writes. Brighter means more.",
  measured:
    "Measured in your browser when the page opened. GPT-2 read 20 random tokens twice, and Featurama checked where each head looked. A dot means the head clearly looks at the token just before, at an earlier copy of the same token, or at the token that followed that copy.",
  research:
    "Found by researchers who traced how GPT-2 completes sentences like “When Mary and John went to the store, John gave a drink to” (Wang et al. 2022, Interpretability in the Wild).",
  score: "From 0 to 1: the share of its attention the head gave to the expected token in the repeated random sequence.",
  attentionRow: "Each row is one token looking back. The shade shows how much of its attention goes to each earlier token; a row adds up to 100%.",
  local: "GPT-2 runs on your graphics card. Nothing you type leaves your browser.",
  guide: "Short hints beside each exhibit about what to try next.",
  shuffle: "Try another starting phrase",
  tab: "Press Tab to add GPT-2's top guess",
} as const;
