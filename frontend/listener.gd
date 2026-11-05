extends Control
## Click -> microphone -> AudioEffectCapture -> WebSocket /ws/recognize -> chord on screen.
## Protocol: first a text frame {"sample_rate": N}, then binary mono Float32 LE frames of HOP_SECONDS
## (backend/src/chordfinder/server.py); every frame gets a StreamReply.

const HOP_SECONDS: float = 0.125  # must match backend stream.HOP_SECONDS: the server's vote counts frames
const NO_AUDIO_TIMEOUT: float = 3.0
const BUS: String = "Capture"

var _ws: WebSocketPeer = WebSocketPeer.new()
var _capture: AudioEffectCapture
var _player: AudioStreamPlayer
var _pending: PackedFloat32Array = PackedFloat32Array()
var _hop: int = 0
var _header_sent: bool = false
var _started_at: float = 0.0
var _status: String = "Нажми, чтобы включить микрофон"
var _shown_status: String = ""
var _shown_stable: String = ""
var _heard: bool = false  # any non-zero sample: a refused or muted mic gives exact zeros, a live one never does

var _button: Button
var _status_label: Label
var _chord_label: Label
var _raw_label: Label
var _bars: Bars
var _measure: Measure  # null unless the page was opened with ?measure
var _record_button: Button
var _redo_button: Button


func _ready() -> void:
	_button = Button.new()
	_button.text = "Включить микрофон"
	_button.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_button.pressed.connect(_start)
	add_child(_button)

	_status_label = _label(22, Control.PRESET_TOP_LEFT)
	_status_label.grow_horizontal = Control.GROW_DIRECTION_END
	_status_label.position = Vector2(16, 16)
	_chord_label = _label(96, Control.PRESET_CENTER_TOP)
	_chord_label.position.y += 40
	_raw_label = _label(22, Control.PRESET_CENTER_BOTTOM)
	_raw_label.position.y -= 40
	_bars = Bars.new()
	_bars.set_anchors_preset(Control.PRESET_FULL_RECT)
	_bars.offset_left = 64
	_bars.offset_right = -64
	_bars.offset_top = 190
	_bars.offset_bottom = -64
	_bars.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_bars)

	if OS.has_feature("web") and "measure" in str(JavaScriptBridge.eval("location.search")):
		_measure = Measure.new()
		_record_button = _measure_button(-336.0, _record)
		_redo_button = _measure_button(-176.0, _redo)
		_redo_button.text = "Заново"


func _label(size: int, preset: Control.LayoutPreset) -> Label:
	var label: Label = Label.new()
	label.add_theme_font_size_override("font_size", size)
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.grow_horizontal = Control.GROW_DIRECTION_BOTH
	label.grow_vertical = Control.GROW_DIRECTION_BOTH
	label.set_anchors_and_offsets_preset(preset)
	add_child(label)
	return label


func _measure_button(left: float, action: Callable) -> Button:
	var button: Button = Button.new()
	button.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	button.offset_left = left
	button.offset_right = left + 150.0
	button.offset_top = 12
	button.offset_bottom = 52
	button.visible = false
	button.pressed.connect(action)
	add_child(button)
	return button


func _start() -> void:
	_button.visible = false
	var bus: int = AudioServer.bus_count
	AudioServer.add_bus(bus)
	AudioServer.set_bus_name(bus, BUS)
	AudioServer.set_bus_mute(bus, true)  # no feedback into the speakers
	_capture = AudioEffectCapture.new()
	_capture.buffer_length = 1.0  # default 0.1 s overflows on a single slow frame
	AudioServer.add_bus_effect(bus, _capture)

	_player = AudioStreamPlayer.new()
	_player.stream = AudioStreamMicrophone.new()
	_player.bus = BUS
	add_child(_player)
	_player.play()
	_started_at = Time.get_ticks_msec() / 1000.0
	_hop = int(AudioServer.get_mix_rate() * HOP_SECONDS)

	_ws.outbound_buffer_size = 1 << 20
	var err: Error = _ws.connect_to_url(_server_url())
	_status = "Соединение…" if err == OK else "Ошибка WebSocket: %s" % error_string(err)


func _server_url() -> String:
	if OS.has_feature("web"):
		var origin: String = str(JavaScriptBridge.eval("location.origin"))
		return origin.replace("http", "ws") + "/ws/recognize"
	return "ws://127.0.0.1:8000/ws/recognize"


func _process(_delta: float) -> void:
	_ws.poll()
	if _capture:
		var n: int = _capture.get_frames_available()
		if n > 0:
			var mono: PackedFloat32Array = Frames.mono(_capture.get_buffer(n))
			if not _heard:
				_heard = Frames.has_sound(mono)
			_pending.append_array(mono)

	match _ws.get_ready_state():
		WebSocketPeer.STATE_OPEN:
			_on_open()
		WebSocketPeer.STATE_CLOSED:
			if _capture:
				_status = "Соединение закрыто (%d)" % _ws.get_close_code()

	if _measure:
		_tick_measure()
	if _capture and not _heard and Time.get_ticks_msec() / 1000.0 - _started_at > NO_AUDIO_TIMEOUT:
		_status = _mic_problem()
	_redraw()


func _on_open() -> void:
	if not _header_sent:
		_ws.send_text(JSON.stringify({"sample_rate": AudioServer.get_mix_rate()}))
		_header_sent = true
		_status = "Играй аккорд"
		if _measure:
			_record_button.visible = true
			_redo_button.visible = true
			_show_measure()
	while _ws.get_available_packet_count() > 0:
		_on_reply(Reply.parse(_ws.get_packet().get_string_from_utf8()))
	# Fixed-size frames, not "whatever piled up": the server votes over consecutive frames.
	while _hop > 0 and _pending.size() >= _hop:
		_ws.send(Frames.pack(_pending.slice(0, _hop)), WebSocketPeer.WRITE_MODE_BINARY)
		_pending = _pending.slice(_hop)


func _on_reply(reply: Reply) -> void:
	if reply.error != "":
		_status = "Ошибка сервера: %s" % reply.error
		return
	if reply.chord == "":
		_shown_stable = ""  # silence ends the chord: the same chord strummed again shows up again
	elif reply.stable != "" and reply.stable != _shown_stable:
		_shown_stable = reply.stable
		print("chord: ", _shown_stable)  # e2e tests read the console
	if _measure and reply.take == _measure.chord():
		_measure.heard(reply.strums, _now())
		_show_measure()
	_chord_label.text = _shown_stable
	_bars.show_reply(reply, _shown_stable)
	_raw_label.text = "тишина · RMS %.3f" % reply.rms if reply.chord == "" else \
		"сейчас: %s · уверенность %.2f · RMS %.3f" % [reply.chord, reply.confidence, reply.rms]


func _now() -> float:
	return Time.get_ticks_msec() / 1000.0


func _take(chord: Variant) -> void:
	_ws.send_text(JSON.stringify({"take": chord}))


func _record() -> void:
	_take(_measure.chord())
	_measure.record()
	_show_measure()


func _redo() -> void:
	if _measure.state == Measure.State.RECORDING:
		_take(null)
	_measure.redo()
	_show_measure()


func _tick_measure() -> void:
	if _measure.tail_over(_now()):
		_take(null)
		_measure.advance()
		_show_measure()


func _show_measure() -> void:
	_status = _measure.text()
	_record_button.text = "Запись %s" % _measure.chord()
	_record_button.disabled = _measure.state != Measure.State.WAITING
	if _measure.state == Measure.State.WAITING:
		_record_button.grab_focus()  # hands are on the guitar: Space presses it


func _mic_problem() -> String:
	var err: String = ""
	if OS.has_feature("web"):
		err = str(JavaScriptBridge.eval("window.cfMicError || ''"))
	match err:
		"NotAllowedError":
			return "Доступ к микрофону запрещён: разреши его в настройках сайта и обнови страницу"
		"NotFoundError":
			return "Микрофон не найден: подключи его и обнови страницу"
		"":
			return "Нет звука с микрофона: проверь, что он включён и не заглушён"
	return "Микрофон недоступен (%s): обнови страницу" % err


func _redraw() -> void:
	if _status != _shown_status:
		_shown_status = _status
		print("status: ", _status)  # e2e tests read the console
	_status_label.text = _status
