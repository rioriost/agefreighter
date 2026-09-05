import { opendir } from "node:fs/promises";
import { join } from "node:path";

/** Shallow and bounded: never follow directory or file symlinks. No bytes upload. */
export async function csvFilesInFolder(path: string): Promise<string[]> {
  const selected: string[] = [];
  let entries = 0;
  for await (const entry of await opendir(path)) {
    if (++entries > 4096) throw new Error("Folder contains too many entries. Select individual CSV files instead.");
    if (!entry.isFile() || !/\.csv$/i.test(entry.name)) continue;
    selected.push(join(path, entry.name));
    if (selected.length > 64) throw new Error("Select a folder with at most 64 CSV files.");
  }
  if (!selected.length) throw new Error("No regular CSV files were found directly in this folder.");
  return selected.sort();
}
