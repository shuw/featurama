// Starting prompts. Some GPT-2 finishes with confidence ("Live long and"
// → " prosper", 95%); others catch it out ("Roses are red, violets are" →
// " green"). Each was checked on the model, but the app always shows what
// the model actually says.
export const PHRASES = [
  "May the Force be with",
  "Once upon a",
  "It was the best of times, it was the worst of",
  "There's no place like",
  "A long time ago in a galaxy far, far",
  "Live long and",
  "With great power comes great",
  "Ask not what your country can do for",
  "Keep calm and carry",
  "To be or not to be, that is the",
  "Winter is",
  "Call me",
  "One small step for man, one giant leap for",
  "Roses are red, violets are",
  "Water boils at 100 degrees",
  "Life is like a box of",
  "I came, I saw, I",
  "Why did the chicken cross the",
  "Never gonna give you",
  "Houston, we have a",
  "I think, therefore I",
  "The first rule of Fight Club is",
  "Twinkle, twinkle, little",
  "Elementary, my dear",
  "I'll be",
];

export function randomPhrase(not?: string) {
  const options = PHRASES.filter((p) => p !== not);
  return options[Math.floor(Math.random() * options.length)];
}
