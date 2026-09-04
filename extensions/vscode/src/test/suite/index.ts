import * as path from "node:path";
import Mocha from "mocha";

export async function run(): Promise<void> {
  const mocha = new Mocha({ ui: "tdd", color: true, timeout: 30_000 });
  mocha.addFile(path.resolve(__dirname, "extension.test.js"));
  await new Promise<void>((resolve, reject) => {
    mocha.run((failures) => failures > 0
      ? reject(new Error(`${failures} Extension Host test(s) failed.`))
      : resolve());
  });
}
