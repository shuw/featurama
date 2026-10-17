// One plain-language explanation per idea, used by every tooltip, so a term
// means the same thing wherever it appears.

export const GLOSSARY = {
  token: "The pieces GPT-2 reads: a word, part of a word, or punctuation. ␣ marks a space at the start of a token.",
  start: "⟨start⟩ is GPT-2's <|endoftext|> token, which Featurama puts in front of every prompt, as researchers do. A head must spread all of its attention over earlier tokens, so when it has nothing useful to fetch it parks it here, where it changes almost nothing. Researchers call this an attention sink.",
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
  shuffle: "Try another starting phrase",
  direct: "What this part's output, on its own, does to GPT-2's next-token scores at this token: the whole words and numbers it makes most and least likely. Later layers can change the final outcome.",
  highlight: "Outline the heads known to do one kind of job.",
  sae: "A sparse autoencoder is a small network trained to rewrite each token's 768 numbers as a few active features out of 24,576, so that each feature tends to mean one thing. This one was trained by Joseph Bloom on web text (MIT licence).",
  feature: "A direction in GPT-2's 768 numbers that a sparse autoencoder found. Each tends to fire on one kind of thing, such as rain, the word “king” or a team sport. Nobody told GPT-2 about these; they were found afterwards.",
  featurePush: "What adding this feature's direction does to the next-token scores on its own, ignoring later layers: the whole words it makes most and least likely.",
  tab: "Press Tab to add GPT-2's top guess",
} as const;
