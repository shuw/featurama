// GPT-2's byte-level BPE tokenizer.

const PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// GPT-2 maps every byte to a printable character so BPE can work on strings.
function byteChars(): string[] {
  const bytes: number[] = [];
  for (let b = 33; b <= 126; b++) bytes.push(b);
  for (let b = 161; b <= 172; b++) bytes.push(b);
  for (let b = 174; b <= 255; b++) bytes.push(b);
  const chars = bytes.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + n++);
    }
  }
  const out: string[] = [];
  bytes.forEach((b, i) => (out[b] = String.fromCodePoint(chars[i])));
  return out;
}

export class Tokenizer {
  private ranks: Map<string, number>;
  private byteToChar = byteChars();
  private charToByte = new Map(this.byteToChar.map((c, b) => [c, b]));
  private idToToken: string[] = [];
  private cache = new Map<string, number[]>();

  constructor(private vocab: Record<string, number>, merges: string) {
    const lines = merges.split("\n").filter((l) => l && !l.startsWith("#version"));
    this.ranks = new Map(lines.map((l, i) => [l, i]));
    for (const [token, id] of Object.entries(vocab)) this.idToToken[id] = token;
  }

  get size() {
    return this.idToToken.length;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    const utf8 = new TextEncoder();
    for (const match of text.matchAll(PATTERN)) {
      const chars = Array.from(utf8.encode(match[0]), (b) => this.byteToChar[b]).join("");
      ids.push(...this.bpe(chars));
    }
    return ids;
  }

  decode(ids: number[]): string {
    return new TextDecoder().decode(this.bytes(ids));
  }

  // How one token reads on its own. Tokens that are part of a multi-byte
  // character show as "�", which is honest: they aren't text by themselves.
  piece(id: number): string {
    return new TextDecoder().decode(this.bytes([id]));
  }

  private bytes(ids: number[]): Uint8Array {
    const out: number[] = [];
    for (const id of ids) for (const c of this.idToToken[id] ?? "") out.push(this.charToByte.get(c)!);
    return new Uint8Array(out);
  }

  private bpe(chars: string): number[] {
    const cached = this.cache.get(chars);
    if (cached) return cached;
    let parts = Array.from(chars);
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best];
      const b = parts[best + 1];
      const merged: string[] = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else merged.push(parts[i]);
      }
      parts = merged;
    }
    const ids = parts.map((p) => {
      const id = this.vocab[p];
      if (id === undefined) throw new Error(`No token for ${JSON.stringify(p)}`);
      return id;
    });
    this.cache.set(chars, ids);
    return ids;
  }
}
