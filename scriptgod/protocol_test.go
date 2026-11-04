package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewMessageValidates(t *testing.T) {
	msg := NewMessage(TASK, "Script.god", "lmlm-code", map[string]any{"instruction":"test"})
	if err := msg.Validate(); err != nil {
		t.Fatalf("expected valid message, got %v", err)
	}
}

func TestMessageRejectsWrongProtocol(t *testing.T) {
	msg := NewMessage(TASK, "Script.god", "lmlm-code", nil)
	msg.Protocol = "GOD/9.9"
	if err := msg.Validate(); err == nil {
		t.Fatal("expected unsupported protocol error")
	}
}

func TestRegisterRequiresIDAndEndpoint(t *testing.T) {
	c := NewCoordinator()
	if err := c.register(Participant{ID:"worker"}); err == nil {
		t.Fatal("expected endpoint validation error")
	}
	if err := c.register(Participant{Endpoint:"http://127.0.0.1:9001"}); err == nil {
		t.Fatal("expected participant ID validation error")
	}
}

func TestUnregisteredSenderCannotDispatch(t *testing.T) {
	c := NewCoordinator()
	req := httptest.NewRequest(http.MethodPost, "/god/v1/messages", strings.NewReader(
		`{"protocol":"GOD/0.1","id":"m1","type":"TASK","sender":"rogue","recipient":"worker"}`,
	))
	rec := httptest.NewRecorder()
	c.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d", rec.Code)
	}
}
