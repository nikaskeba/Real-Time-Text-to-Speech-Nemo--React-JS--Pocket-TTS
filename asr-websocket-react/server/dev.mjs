import { spawn } from "node:child_process";

const children = [
  spawn("npm", ["run", "dev:proxy"], {
    stdio: "inherit",
    shell: true,
  }),
  spawn("npm", ["run", "dev:client"], {
    stdio: "inherit",
    shell: true,
  }),
];

const stop = (code = 0) => {
  for (const child of children) {
    if (!child.killed) {
      child.kill("SIGTERM");
    }
  }
  process.exit(code);
};

for (const child of children) {
  child.on("exit", (code) => {
    if (code && code !== 0) {
      stop(code);
    }
  });
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));
