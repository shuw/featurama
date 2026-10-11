// Hand-written names for features we've read closely, from their strongest
// passages and the words they push toward. Unlisted features are named by
// the tokens they fire on most.

export const LABELS: Record<number, string> = {
  5531: "Pirates",
  7626: "Rain",
  4491: "Team sports",
  7529: "Elementary school",
  23996: "“and” after police: emergency services",
  24120: "Planets",
  2447: "PC games",
  5383: "“and” between sciences",
  18414: "Pro soccer leagues",
  2173: "Food and eating",
  8243: "Dogs",
  16767: "Kings",
  20174: "The ocean",
  17582: "The Pacific",
};

// Features that steer GPT-2's writing clearly at the default strength, from
// trying each on a few prompts.
export const STEER_PICKS: number[] = [5531, 7626, 24120, 7529, 2447, 2173, 4491, 23996, 8243];
