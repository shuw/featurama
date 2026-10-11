import { expect, test } from "bun:test";
import { tokenizer, tokenizerCases } from "./helpers.ts";

test("matches the Hugging Face GPT-2 tokenizer", async () => {
  const tok = await tokenizer();
  for (const c of await tokenizerCases()) {
    expect({ text: c.text, ids: tok.encode(c.text) }).toEqual(c);
    expect(tok.decode(c.ids)).toBe(c.text);
  }
});

test("splits words the way the model sees them", async () => {
  const tok = await tokenizer();
  expect(tok.encode(" Eiffel").map((id) => tok.piece(id))).toEqual([" E", "iff", "el"]);
});
