# Script.god runtime

Standalone Go HTTP coordinator for the LMLM GOD/0.1 protocol.

## Scripts

- `../Script.god` — root coordination manifest.
- `../scripts/Script.dm` — direct messaging contract.
- `../scripts/Script.byod` — bring-your-own-data ingestion contract.
- `../scripts/Script.sync` — versioned knowledge propagation contract.
- `../scripts/Script.policy` — default-deny permissions.
- `../scripts/Script.verify` — verification contract.

These files are independent script manifests. They share the GOD protocol but retain separate responsibilities.

## Run

```sh
cd scriptgod
go run .
```

The coordinator listens on `127.0.0.1:8787` by default. Set `GOD_LISTEN` to override it.

## HTTP API

- `GET /healthz`
- `GET /god/v1/participants`
- `POST /god/v1/register`
- `POST /god/v1/messages`

Register a model/script participant:

```sh
curl -X POST http://127.0.0.1:8787/god/v1/register \
  -H 'Content-Type: application/json' \
  -d '{"id":"lmlm-code","endpoint":"http://127.0.0.1:9001","capabilities":["coding","cuda"]}'
```

Dispatch a task:

```sh
curl -X POST http://127.0.0.1:8787/god/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"protocol":"GOD/0.1","id":"task-001","type":"TASK","sender":"Script.god","recipient":"lmlm-code","timestamp":"2026-11-04T00:00:00Z","payload":{"instruction":"Review this task"}}'
```

A participant endpoint must implement `POST /god/v1/messages` and return an HTTP success response after accepting the message.

## Security boundary

The initial runtime binds to loopback by default, limits request bodies, requires registered senders/recipients, and restricts dispatch of TASK, CONTEXT, DATA, and SYNC messages to Script.god. The policy manifests are specifications; full policy evaluation, authentication/signatures, persistent storage, retries, and independent verification are not implemented yet. Do not expose this unauthenticated prototype to an untrusted network.
