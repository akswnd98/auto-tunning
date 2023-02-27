from vpython import *

ball = sphere(
  pos=vec(-10, 0, 0),
  texture=textures.rock,
  make_trail=True,
  trail_type='points',
  trail_radius=0.05,
  interval=4
)

ball.v = vec(4, 0, 0)
t = 0
dt = 0.01

while t < 5:
  rate(1 / dt)
  ball.pos = ball.pos + ball.v * dt
  t = t + dt
