# Use a Full-Screen Terminal Conversation

Running `myclaw` without arguments replaces the plain scrolling REPL with a full-screen terminal UI containing a conversation display and a bottom input area. This gives MyClaw stable ownership of message alignment, scrolling, streaming updates, input state, operational events, and Tool Confirmation at the cost of shell-native scrollback and greater responsibility for restoring the terminal after cancellation, failure, or exit; maintaining a second interactive REPL is deliberately out of scope.
