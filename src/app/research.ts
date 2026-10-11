// What published research says about GPT-2 small's attention heads, in our
// own plain words, with the source for each claim. Head ids are layer.head.

export const SOURCES = {
  every: {
    name: "Krzyzanowski, Kissane, Conmy & Nanda (2024), “We Inspected Every Head in GPT-2 Small”; also Kissane et al. (2024), arXiv 2406.17759",
    url: "https://arxiv.org/abs/2406.17759",
  },
  ioi: { name: "Wang et al. (2022), Interpretability in the Wild (the Mary-and-John study)", url: "https://arxiv.org/abs/2211.00593" },
  greater: { name: "Hanna, Liu & Variengien (2023), How does GPT-2 compute greater-than?", url: "https://arxiv.org/abs/2305.00586" },
  copy: { name: "McDougall et al. (2023), Copy Suppression", url: "https://arxiv.org/abs/2310.04625" },
  sequence: { name: "Lan, Torr & Barez (2023), Interpretable Sequence Continuation", url: "https://arxiv.org/abs/2311.04131" },
  acronym: { name: "García-Carrasco, Maté & Trujillo (2024), How does GPT-2 Predict Acronyms?", url: "https://arxiv.org/abs/2405.04156" },
  pronoun: { name: "Mathwin et al. (2023), a preliminary circuit for gendered pronouns (hackathon report, not peer reviewed)", url: "https://arxiv.org/abs/2304.14997" },
  attnsae: { name: "Kissane et al. (2024), Interpreting Attention Layer Outputs with Sparse Autoencoders", url: "https://arxiv.org/abs/2406.17759" },
} as const;
export type SourceKey = keyof typeof SOURCES;

// How many separate jobs the "every head" study found. Most heads do several.
export const BREADTH = {
  mono: "one clear job",
  mostly: "mostly one job",
  two: "two jobs",
  many: "many jobs",
  several: "several jobs, like most heads",
} as const;
type Breadth = keyof typeof BREADTH;

// The "every head" study's finding for each head, reworded for non-specialists.
const STUDY: Record<string, [string, Breadth] | null> = {
  "0.0": ["Reacts to particular words on their own.", "several"],
  "0.1": ["Reacts to “of” and to digits.", "several"],
  "0.2": ["Reacts to related word pairs and single words.", "several"],
  "0.3": ["Reacts to single words and short word pairs.", "several"],
  "0.4": ["Barely active in the study.", "several"],
  "0.5": ["Finds earlier copies of the current word; also word endings.", "several"],
  "0.6": ["Tracks headline-style text and punctuation.", "several"],
  "0.7": ["Recognises familiar word pairs, like “Big Bang”.", "several"],
  "0.8": ["Tracks the topic of the last few words, such as cars or gadgets.", "several"],
  "0.9": ["Tracks the style of the whole passage, such as headlines or lists.", "several"],
  "0.10": ["Reacts to single words, years and lists of places.", "several"],
  "0.11": ["Reacts to words like “not” and to first, second, third.", "several"],
  "1.0": ["Completes pairs that start with a capital letter; units of time.", "several"],
  "1.1": ["Reacts to named organisations, social-media handles and pronouns.", "several"],
  "1.2": ["Tracks headline-style text; completes word endings.", "several"],
  "1.3": ["Barely active; some news context.", "several"],
  "1.4": ["Tracks the topic of the whole passage, such as news or web addresses.", "several"],
  "1.5": ["Recognises things that come in order (one, two, three; Monday, Tuesday).", "mono"],
  "1.6": ["Reacts to verbs and what they describe.", "several"],
  "1.7": ["Mostly unclear; some news context.", "several"],
  "1.8": ["Too weak for researchers to pin down.", "several"],
  "1.9": ["Researchers couldn't tell what it does.", "several"],
  "1.10": ["A few topic signals; full stops.", "several"],
  "1.11": ["Finds earlier copies of the current word.", "several"],
  "2.0": ["Handles short phrases after “not”, “just” or “never”.", "several"],
  "2.1": ["Reacts to political parties and institutions.", "several"],
  "2.2": ["Joins word pieces into words and short phrases.", "several"],
  "2.3": ["Handles what follows amounts, times and places.", "mono"],
  "2.4": ["Reacts to particular words on their own.", "several"],
  "2.5": ["Handles directions (up, down) and logic words (because, so).", "several"],
  "2.6": ["Tracks news context; pairs of pronouns.", "several"],
  "2.7": ["Weak; some news and Bible context.", "several"],
  "2.8": ["Recognises idioms and money phrases.", "several"],
  "2.9": ["Links a person or thing to what it did.", "mono"],
  "2.10": ["General-purpose patterns; news and sport.", "several"],
  "2.11": ["Mostly rests on the start token; single words.", "several"],
  "3.0": ["Finds earlier copies of the current word, strongly.", "several"],
  "3.1": ["Handles “either… or” and “not only… but also”.", "several"],
  "3.2": ["Handles what follows “so”, “such” or “how”.", "mono"],
  "3.3": ["Joins word fragments into whole words.", "mostly"],
  "3.4": ["Reacts to nationalities, public figures and gender.", "several"],
  "3.5": ["Handles amounts and comparisons (more, fewer, most).", "several"],
  "3.6": ["Reacts to government words, extremes and sizes.", "several"],
  "3.7": ["Single words; phrases about work to be done.", "several"],
  "3.8": ["Reacts to action verbs and groups of people.", "several"],
  "3.9": ["Handles linking words after commas, like “while” and “though”.", "several"],
  "3.10": ["Mixed; may spot repeats.", "several"],
  "3.11": ["Handles order and extremes (first, whole, most).", "several"],
  "4.0": ["Reacts to action verbs, and to amounts.", "two"],
  "4.1": ["Handles “why…” questions, and closeness.", "several"],
  "4.2": ["Keeps track of news context.", "mono"],
  "4.3": ["Handles “should”-type statements, and named things.", "two"],
  "4.4": ["Reacts to dates and amounts, and to platforms.", "two"],
  "4.5": ["Handles what's typical and what's extreme.", "several"],
  "4.6": ["Reacts to pronouns, including he and she.", "several"],
  "4.7": ["Finds earlier copies of words, in an unusual way.", "mono"],
  "4.8": ["Researchers couldn't tell what it does.", "several"],
  "4.9": ["Handles phrases about spans of time.", "several"],
  "4.10": ["May look at the word before (unconfirmed).", "several"],
  "4.11": ["Looks at the word just before.", "mono"],
  "5.0": ["Single words; web addresses.", "several"],
  "5.1": ["Induction: spots a repeat and predicts what came next last time.", "several"],
  "5.2": ["Superlatives (biggest, best) and legal phrases.", "several"],
  "5.3": ["Set phrases like “of course”; page footers; news.", "several"],
  "5.4": ["Particular named things; “at least”.", "several"],
  "5.5": ["Induction: spots a repeat and predicts what came next last time.", "several"],
  "5.6": ["Looks at the word just before.", "several"],
  "5.7": ["Action verbs; “between X and Y”.", "several"],
  "5.8": ["Mostly single words.", "several"],
  "5.9": ["Pronouns and small words at the start of a sentence.", "several"],
  "5.10": ["Tracks named things nearby.", "several"],
  "5.11": ["Single words; legal text.", "several"],
  "6.0": ["What follows a pronoun; political figures.", "several"],
  "6.1": ["Idioms like “time and effort”.", "several"],
  "6.2": ["Some long-range topic signal; unclear.", "several"],
  "6.3": ["Verbs after a comma; “just” and “ask”.", "mostly"],
  "6.4": ["News context.", "several"],
  "6.5": ["Phrases about reaching agreement.", "several"],
  "6.6": ["Mostly rests on the start token; endings of web addresses.", "several"],
  "6.7": ["The topic of the last few words.", "several"],
  "6.8": null,
  "6.9": ["Induction: spots a repeat and predicts what came next last time.", "mono"],
  "6.10": ["“X said” and “X replied”.", "several"],
  "6.11": ["Completes the endings of verbs and phrases.", "mostly"],
  "7.0": ["Many jobs, including “as… as” comparisons.", "several"],
  "7.1": ["Action verbs; “and” and “or”; pronouns.", "several"],
  "7.2": ["An unusual kind of induction.", "mono"],
  "7.3": ["General-purpose patterns; “come in”, “come back”.", "several"],
  "7.4": null,
  "7.5": ["Family relations (mother, brother), among many other things.", "many"],
  "7.6": ["Line breaks; long-range topic.", "several"],
  "7.7": ["Pairs joined by “and”, like “salt and pepper”.", "several"],
  "7.8": ["How things are going; actions taken.", "several"],
  "7.9": ["Reasons and justifications.", "mostly"],
  "7.10": ["Induction: spots a repeat and predicts what came next last time.", "mono"],
  "7.11": ["To do with how words are split into tokens.", "several"],
  "8.0": ["News snippets.", "several"],
  "8.1": ["Copies small linking words like “in”, “on” and “of”.", "mono"],
  "8.2": ["Standard news phrasing.", "several"],
  "8.3": null,
  "8.4": ["Action verbs like prevent, provide and hold.", "several"],
  "8.5": ["Pairs like “either/or” and “neither/nor”.", "several"],
  "8.6": ["Completes web addresses.", "several"],
  "8.7": ["Many different things.", "several"],
  "8.8": ["Comparing amounts and times.", "several"],
  "8.9": ["Phrases like “more than you think”; answering why.", "several"],
  "8.10": ["Ranges of ages and money.", "several"],
  "8.11": null,
  "9.0": ["Keeps track of complex ideas.", "mono"],
  "9.1": ["Successor: predicts what comes next in a sequence (Monday → Tuesday).", "several"],
  "9.2": ["Completes complex ideas.", "mostly"],
  "9.3": ["Completes quotes; “more X than”.", "several"],
  "9.4": ["Small words after pronouns.", "several"],
  "9.5": ["“the X-year-old”.", "several"],
  "9.6": ["Brackets and capitalised names.", "several"],
  "9.7": ["“Fits into”; smallness.", "several"],
  "9.8": ["News context.", "several"],
  "9.9": ["Copies earlier words, usually names.", "mono"],
  "9.10": ["Joining words like “from… to”.", "several"],
  "9.11": ["Researchers couldn't tell what it does.", "several"],
  "10.0": ["Political names.", "several"],
  "10.1": ["Physical and spatial properties.", "several"],
  "10.2": ["Completes web addresses and copies digits.", "several"],
  "10.3": null,
  "10.4": ["Separators, like the colon in a time or the dash in a phone number.", "several"],
  "10.5": ["“What if” words and tenses.", "mostly"],
  "10.6": ["Official titles, like President or Senator.", "mostly"],
  "10.7": ["Copy suppression: stops the model repeating a word it has already seen.", "several"],
  "10.8": null,
  "10.9": ["Relations like “with”, “for” and “from”.", "several"],
  "10.10": ["Completes words that start with a capital letter.", "mono"],
  "10.11": ["How ideas relate to each other.", "two"],
  "11.0": ["Political subjects.", "several"],
  "11.1": ["Very rarely active.", "several"],
  "11.2": ["Unusual topic tracking.", "several"],
  "11.3": ["Long-range topic; may adjust how confident the model is.", "mostly"],
  "11.4": ["Completes words that start with a capital letter.", "several"],
  "11.5": ["“can”, “be”, “done”, “will”.", "several"],
  "11.6": ["Rare two-word completions.", "several"],
  "11.7": ["Small grammar fixes.", "several"],
  "11.8": ["Small grammar fixes.", "several"],
  "11.9": ["Particular two-word completions.", "several"],
  "11.10": null,
  "11.11": null,
};

// Roles found in studies of specific tasks.
const IOI_ROLE_TEXT = {
  duplicate: "Notices that a name has already appeared.",
  previous: "Passes each word on to the next position, which induction heads build on.",
  induction: "Spots the repeated name by induction.",
  inhibition: "Tells the name movers which name to skip.",
  mover: "Copies the right name into the answer.",
  backup: "Takes over copying the name if the name movers are switched off.",
  negative: "Pushes against the name the name movers pick.",
};
const YEAR = "In “from 1732 to 17…”, reads the start year so the model can pick a later one.";
const LETTER = "Writes the right capital letter when finishing an acronym.";

const CIRCUITS: Record<string, { source: SourceKey; role: string; text: string }[]> = {};
const add = (ids: string[], source: SourceKey, role: string, text: string) => {
  for (const id of ids) (CIRCUITS[id] ??= []).push({ source, role, text });
};
add(["0.1", "3.0"], "ioi", "Duplicate token", IOI_ROLE_TEXT.duplicate);
add(["0.10"], "ioi", "Duplicate token (weaker)", IOI_ROLE_TEXT.duplicate);
add(["2.2", "4.11"], "ioi", "Previous token", IOI_ROLE_TEXT.previous);
add(["5.5", "6.9"], "ioi", "Induction", IOI_ROLE_TEXT.induction);
add(["5.8", "5.9"], "ioi", "Induction (weaker)", IOI_ROLE_TEXT.induction);
add(["7.3", "7.9", "8.6", "8.10"], "ioi", "Name inhibitor", IOI_ROLE_TEXT.inhibition);
add(["9.6", "9.9", "10.0"], "ioi", "Name mover", IOI_ROLE_TEXT.mover);
add(["9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"], "ioi", "Backup name mover", IOI_ROLE_TEXT.backup);
add(["10.7", "11.10"], "ioi", "Negative name mover", IOI_ROLE_TEXT.negative);
add(["5.1", "5.5", "6.9", "7.10", "8.8", "8.11", "9.1"], "greater", "Year reader", YEAR);
add(["10.7"], "copy", "Copy suppression", "Stops the model repeating a word that's already in the text: the main head that does this.");
add(["11.10"], "copy", "Copy suppression", "A second, weaker copy-suppression head that backs up 10.7.");
add(["9.1"], "sequence", "Successor", "Predicts the next item in a sequence: 1 → 2, January → February.");
add(["1.5"], "sequence", "Same-kind finder", "Links items of the same kind in a sequence.");
add(["4.4"], "sequence", "Neighbour finder", "Links neighbouring items in an ordered sequence.");
add(["7.11"], "sequence", "Sequence mover", "Brings what it knows about the sequence to the last word.");
add(["0.1"], "sequence", "Needed for sequences", "Switching it off makes continuing a sequence more than 20% worse.");
add(["1.0", "2.2", "4.11"], "acronym", "Letter passer", "Passes the previous capital letter forward while an acronym is written.");
add(["5.8"], "acronym", "Letter mover", "Moves capital-letter information to where the acronym is written.");
add(["8.11", "9.9", "10.10", "11.4"], "acronym", "Letter writer", LETTER);
add(["0.4"], "pronoun", "Name reader", "Reads the name early, so he or she can be chosen later.");
add(["4.3", "6.0"], "pronoun", "Name finder", "Looks back at the name to help choose he or she.");
add(["9.7", "10.9"], "pronoun", "Pronoun writer", "Writes he or she.");
add(["5.1"], "attnsae", "Long-match induction", "Does induction when two or more words repeat.");
add(["5.5"], "attnsae", "Short-match induction", "Does induction when a single word repeats.");
add(["10.2"], "attnsae", "Two jobs, confirmed", "Copies digits and completes web addresses; switching it off confirmed both.");

// What studies found about particular layers' MLPs.
export const MLP_NOTES: Record<number, { source: SourceKey; role: string; text: string }[]> = {
  0: [
    { source: "copy", role: "Part of the embedding", text: "Works like an extension of the word embedding: researchers often treat it as part of how each word is first represented." },
    { source: "pronoun", role: "Pronoun pathway", text: "One of the MLPs on the path that picks he or she." },
  ],
  3: [{ source: "pronoun", role: "Pronoun pathway", text: "One of the MLPs on the path that picks he or she." }],
  5: [{ source: "pronoun", role: "Pronoun pathway", text: "One of the MLPs on the path that picks he or she." }],
  7: [{ source: "pronoun", role: "Pronoun pathway", text: "One of the MLPs on the path that picks he or she." }],
  8: [{ source: "greater", role: "Year comparison", text: "Helps work out which years are later than the start year, mostly by feeding the MLPs after it." }],
  9: [
    { source: "greater", role: "Later years", text: "Raises the odds of years later than the start year." },
    { source: "sequence", role: "Sequences", text: "Critical for continuing sequences: switching it off makes them over 90% worse." },
  ],
  10: [
    { source: "greater", role: "Later years", text: "Raises the odds of years later than the start year." },
    { source: "pronoun", role: "Pronoun writer", text: "Has one of the largest direct effects on he or she." },
  ],
  11: [
    { source: "greater", role: "Caps the range", text: "Keeps the predicted year within about 50 years of the start." },
    { source: "pronoun", role: "Pronoun writer", text: "Has one of the largest direct effects on he or she." },
  ],
};

// Prompts that should wake up a few well-understood heads.
const REPEAT = "The cat sat on the mat. The cat sat on the";
const NAMES = "When Mary and John went to the store, John gave a drink to";
export const TRY_PROMPTS: Record<string, string> = {
  "9.1": "Monday, Tuesday, Wednesday, Thursday,",
  "1.5": "one, two, three, four,",
  "5.1": REPEAT, "5.5": REPEAT, "6.9": REPEAT, "7.2": REPEAT, "7.10": REPEAT,
  "4.11": "Once upon a time there was a",
  "2.2": "Once upon a time there was a",
  "9.6": NAMES, "9.9": NAMES, "10.0": NAMES, "10.7": NAMES, "11.10": NAMES, "8.6": NAMES, "7.3": NAMES, "7.9": NAMES, "8.10": NAMES,
  "0.1": NAMES, "3.0": NAMES,
  "8.11": "The Federal Bureau of Investigation (F", "10.10": "The Federal Bureau of Investigation (F", "11.4": "The Federal Bureau of Investigation (F",
  "8.8": "The war lasted from the year 1732 to the year 17",
  "9.7": "Mary went to the shop because", "10.9": "Mary went to the shop because",
};

// The every-head survey's online dashboard for one head.
export const dashboardUrl = (id: string) => `https://robertzk.github.io/gpt2-small-saes/cards/top_features_${id.replace(".", "_")}.html`;

// A head's one-line label: the survey's, or else what we measured.
export function label(id: string) {
  return STUDY[id]?.[0] ?? null;
}

export function research(id: string) {
  const study = STUDY[id];
  return {
    study: study ? { text: study[0], breadth: BREADTH[study[1]] } : null,
    circuits: CIRCUITS[id] ?? [],
  };
}
