class Gaze_Classifier:
    def __init__(self, cx, cy, dead_zone=0.08, confirm_frames=6):
        self.cx            = cx
        self.cy            = cy
        self.dead_zone     = dead_zone
        self.confirm_frames = confirm_frames
        self.history       = []
        self.confirmed     = ("CENTER", "CENTER")

    def update(self, nx, ny):
        dx = nx - self.cx
        # dy = ny - self.cy

        h = "LEFT"   if dx < -self.dead_zone else \
            "RIGHT"  if dx >  self.dead_zone else "CENTER"
        # v = "UP"     if dy < -self.dead_zone else \
        #     "DOWN"   if dy >  self.dead_zone else "CENTER"
        
        self.history.append(h)
        if len(self.history) > self.confirm_frames:
            self.history.pop(0)

        # only confirm a direction if it's consistent across all recent frames
        if all(d == h for d in self.history):
            self.confirmed = (h, self.confirmed[1])
        # if all(d[1] == v for d in self.history):
        #     self.confirmed = (self.confirmed[0], v)

        return self.confirmed