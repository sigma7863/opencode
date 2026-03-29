import { randomBytes } from "node:crypto"
import os from "node:os"
import { Server } from "../../server/server"
import { cmd } from "./cmd"
import { withNetworkOptions, resolveNetworkOptions } from "../network"
import { Flag } from "../../flag/flag"
import { Workspace } from "../../control-plane/workspace"
import { Project } from "../../project/project"
import { Installation } from "../../installation"
import { PushRelay } from "../../server/push-relay"
import * as QRCode from "qrcode"

function hosts(hostname: string, port: number) {
  const list = new Set<string>()
  const add = (item: string) => {
    if (!item) return
    if (item === "0.0.0.0") return
    if (item === "::") return
    list.add(`http://${item}:${port}`)
  }
  add(hostname)
  add("127.0.0.1")
  Object.values(os.networkInterfaces())
    .flatMap((item) => item ?? [])
    .filter((item) => item.family === "IPv4" && !item.internal)
    .map((item) => item.address)
    .forEach(add)
  return [...list]
}

export const ServeCommand = cmd({
  command: "serve",
  builder: (yargs) =>
    withNetworkOptions(yargs)
      .option("relay-url", {
        type: "string",
        describe: "experimental APN relay URL",
      })
      .option("relay-secret", {
        type: "string",
        describe: "experimental APN relay secret",
      }),
  describe: "starts a headless opencode server",
  handler: async (args) => {
    if (!Flag.OPENCODE_SERVER_PASSWORD) {
      console.log("Warning: OPENCODE_SERVER_PASSWORD is not set; server is unsecured.")
    }
    const opts = await resolveNetworkOptions(args)
    const server = await Server.listen(opts)
    console.log(`opencode server listening on http://${server.hostname}:${server.port}`)

    const relayURL = (
      args["relay-url"] ??
      process.env.OPENCODE_EXPERIMENTAL_PUSH_RELAY_URL ??
      "https://apn.dev.opencode.ai"
    ).trim()
    const input = (args["relay-secret"] ?? process.env.OPENCODE_EXPERIMENTAL_PUSH_RELAY_SECRET ?? "").trim()
    const relaySecret = input || randomBytes(18).toString("base64url")
    if (!input) {
      console.log("experimental push relay secret generated")
    }
    if (relayURL && relaySecret) {
      const host = server.hostname ?? opts.hostname
      const port = server.port || opts.port || 4096
      const started = PushRelay.start({
        relayURL,
        relaySecret,
        hostname: host,
        port,
      })
      const pair = started ??
        PushRelay.pair() ?? {
          v: 1 as const,
          relayURL,
          relaySecret,
          hosts: hosts(host, port),
        }
      if (!started) {
        console.log("experimental push relay failed to initialize; showing setup qr anyway")
      }
      if (pair) {
        console.log("experimental push relay enabled")
        const payload = JSON.stringify(pair)
        const code = await QRCode.toString(payload, {
          type: "terminal",
          small: true,
          errorCorrectionLevel: "M",
        })
        console.log("scan qr code in mobile app")
        console.log(code)
        console.log("qr payload")
        console.log(JSON.stringify(pair, null, 2))
      }
    }

    await new Promise(() => {})
    await server.stop()
  },
})
