# Arcby — Raspberry Pi 4 GPIO allocation (grouped layout, 2026-10-28)

Physical pin numbers. Pin 1 = 3.3 V corner nearest the SD card; inner row = odd pins, outer row = even pins.
Each module's wires sit together on the header so they are easy to plug.
Fixed by hardware: I2S audio only on GPIO18/19/20/21, I2C (pan-tilt) only on GPIO2/3, only two 5 V pins (2, 4) and two 3.3 V pins (1, 17).

| Group | Pin | GPIO | Connects to |
|---|---|---|---|
| I2C bus (shared) | 3 | GPIO2 SDA | breadboard A2 LEFT -> MPU6050 SDA + PTZ board SDA |
| | 5 | GPIO3 SCL | breadboard A3 LEFT -> MPU6050 SCL + PTZ board SCL |
| Pan-tilt (PTZ board, 4-wire cable) | - | - | VCC + GND from the **4.7 V servo buck** via breadboard A1/A2 RIGHT: the single VCC also feeds the servos, never use Pi 3.3 V. SDA/SCL see I2C bus |
| IMU (MPU6050) + HC-SR04 power | 1 | 3.3V | breadboard A4 LEFT (3.3 V) -> MPU VCC (~4 mA) + HC-SR04 VCC (~15 mA) |
| | 25 | GND | MPU GND (direct) |
| Fan | 4 | 5V | fan + |
| | 6 | GND | fan − |
| | 7 | GPIO4 | fan control (blue wire), on/off by the kernel: `dtoverlay=gpio-fan,gpiopin=4,temp=60000` |
| Motors (L298N) — L298N header order; left motor inner row, right motor outer row | 11 | GPIO17 | ENA (cap removed; signal pin in line with IN1–IN4) |
| | 13 | GPIO27 | IN1 |
| | 15 | GPIO22 | IN2 |
| | 16 | GPIO23 | IN3 |
| | 18 | GPIO24 | IN4 |
| | 22 | GPIO25 | ENB (cap removed) |
| | 14 | GND | L298N GND screw terminal (common ground — mandatory) |
| Distance sensor (HC-SR04) | 29 | GPIO5 | TRIG (direct) |
| | 31 | GPIO6 | ECHO (direct - safe because the sensor runs on 3.3 V; no divider) |
| | 30 | GND | HC-SR04 GND (direct). VCC: see pin 1 (3.3 V via breadboard col 4) |
| Audio (I2S) | 12 | GPIO18 | BCLK -> breadboard A11 (amp BCLK) + mic SCK |
| | 35 | GPIO19 | LRCLK -> breadboard A12 (amp LRC) + mic WS |
| | 40 | GPIO21 | amp DIN (breadboard A10) |
| | 38 | GPIO20 | mic SD (direct) |
| | 2 | 5V | amp VIN (breadboard A6) |
| | 34 | GND | amp GND (breadboard A7) |
| | 17 | 3.3V | mic VDD (direct) |
| | 39 | GND | mic GND (direct; mic L/R tied to its own GND pin) |

Free: 8, 9, 10, 19, 20, 21, 23, 24, 26, 32, 33, 36*, 37. Avoid 27/28 (HAT EEPROM); *36 = GPIO16, claimed by the audio overlay.

**Boot state rule:** GPIO0–8 have pull-ups and are HIGH while the Pi boots; GPIO9–27 are pull-down (LOW).
Never put motor-driver inputs (IN/EN) on GPIO0–8 — the wheels would spin at power-up (happened with GPIO7/8).

Fan: the Armor Lite sheet says blue wire -> GPIO14 (pin 8), but GPIO14 is the serial-console TX
(`enable_uart=1`, console on serial0) and would toggle the fan. Moved to GPIO4 (pin 7, boot HIGH = fan
runs briefly at boot, harmless). The `gpio-fan` overlay switches it on above 60 C, off ~10 C lower.

Shared signals go through the mini breadboard under the upper deck: see `wiring.md` -> Breadboard.

## Audio config (`/boot/firmware/config.txt`)
```
#dtparam=audio=on
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```
Tests: `aplay -l`, `arecord -l`, `speaker-test -c2 -t wav -l 1`,
`arecord -D plughw:0 -c2 -r 48000 -f S32_LE -d 5 test.wav && aplay test.wav` (card number may differ).
Applied to config.txt 2026-10-30 (`dtparam=audio=on` commented out); speaker/mic test pending the reboot.

## History
- 2026-10-28: first drive test used GPIO5/6/12/13/23/26. Re-grouped to pins 19–26 (GPIO10/9/11/25/8/7) — left wheel spun at boot
  because GPIO7/8 boot HIGH. Final: GPIO17/27/22/23/24/25 (pins 11/13/15/16/18/22), GND 14. Distance sensor on GPIO5/6.
