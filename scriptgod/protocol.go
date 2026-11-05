package main

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"time"
)

const ProtocolVersion = "GOD/0.1"

type MessageType string

const (
	CONNECT MessageType = "CONNECT"
	CAPABILITIES MessageType = "CAPABILITIES"
	TASK MessageType = "TASK"
	ACK MessageType = "ACK"
	CONTEXT MessageType = "CONTEXT"
	PROGRESS MessageType = "PROGRESS"
	RESULT MessageType = "RESULT"
	ERROR MessageType = "ERROR"
	BLOCKED MessageType = "BLOCKED"
	CANCEL MessageType = "CANCEL"
	VERIFY MessageType = "VERIFY"
	SYNC MessageType = "SYNC"
	DATA MessageType = "DATA"
)

type Message struct {
	Protocol string         `json:"protocol"`
	ID string               `json:"id"`
	CorrelationID string    `json:"correlation_id,omitempty"`
	Type MessageType        `json:"type"`
	Sender string           `json:"sender"`
	Recipient string        `json:"recipient"`
	Timestamp time.Time     `json:"timestamp"`
	Payload map[string]any  `json:"payload,omitempty"`
	Context map[string]any  `json:"context,omitempty"`
	Constraints []string    `json:"constraints,omitempty"`
}

func NewMessage(kind MessageType, sender, recipient string, payload map[string]any) Message {
	return Message{
		Protocol: ProtocolVersion,
		ID: newID(),
		Type: kind,
		Sender: sender,
		Recipient: recipient,
		Timestamp: time.Now().UTC(),
		Payload: payload,
	}
}

func (m Message) Validate() error {
	if m.Protocol != ProtocolVersion { return errors.New("unsupported GOD protocol version") }
	if m.ID == "" || m.Sender == "" || m.Recipient == "" { return errors.New("id, sender and recipient are required") }
	if m.Type == "" { return errors.New("message type is required") }
	return nil
}

func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil { panic(err) }
	return hex.EncodeToString(b[:])
}
