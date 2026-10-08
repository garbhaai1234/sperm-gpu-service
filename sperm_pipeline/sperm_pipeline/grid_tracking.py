"""Read-only canonical tracking diagnostics. Never feeds observations back to tracking."""
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class GridConfig:
    show_tracking_grid: bool = False
    rows: int = 8
    columns: int = 8
    history_length: int = 40  # source frames, not number of observations
    jump_speed_px_s: float = 300.0  # advisory only, NOT a tracker gate
    nearby_distance_px: float = 40.0
    nearby_gap_seconds: float = 1.0

    def __post_init__(self):
        for name in ('rows', 'columns', 'history_length'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('jump_speed_px_s', 'nearby_distance_px', 'nearby_gap_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')


def bbox_center(bbox):
    """Same xyxy center as uploaded CASA. Never substitute a mask centroid."""
    x1, y1, x2, y2 = map(float, bbox)
    return (x1 + x2) / 2, (y1 + y2) / 2


def column_name(index):
    name = ''
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def grid_cell(x, y, width, height, config):
    # Invalid/out-of-image predictions are not silently clamped into visible cells.
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
        return None, None, None
    col = min(int(x / (width / config.columns)), config.columns - 1)
    row = min(int(y / (height / config.rows)), config.rows - 1)
    return row, col, f'{column_name(col)}{row + 1}'


class GridDiagnostics:
    """Copies finalized links/state each frame. Own history is for rendering only."""
    def __init__(self, job_id, width, height, fps, config=None):
        self.config = config or GridConfig()
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Grid requires positive source dimensions and FPS')
        self.job_id, self.width, self.height, self.fps = str(job_id), width, height, fps
        self.records, self.events = [], []
        self.last = {}
        self.velocity = {}
        self.last_frame = -1
        self.event_offsets = defaultdict(int)

    def _row(self, frame, app_id=None, byte_id=None, bbox=None, point=None,
             confidence=None, state='UNRESOLVED', kind='observed', assignment=None,
             groups=None, source='bbox_center', reason=None):
        x, y = point if point is not None else (None, None)
        row, col, cell = (grid_cell(x, y, self.width, self.height, self.config)
                          if x is not None else (None, None, None))
        previous = self.last.get(app_id) if app_id is not None else None
        distance = (math.hypot(x - previous['x'], y - previous['y'])
                    if previous and x is not None else None)
        return dict(job_id=self.job_id, frame_index=frame, timestamp_seconds=frame / self.fps,
                    application_id=app_id, byte_track_id=byte_id, x=x, y=y,
                    grid_row=row, grid_column=col, grid_cell=cell,
                    detection_confidence=confidence, bbox=list(map(float, bbox)) if bbox is not None else None,
                    identity_state=state, coordinate_kind=kind, coordinate_source=source,
                    in_frame=cell is not None, previous_grid_cell=previous['grid_cell'] if previous else None,
                    movement_distance=distance, assignment_type=assignment,
                    overlap_group_id=groups[0] if groups and len(groups) == 1 else None,
                    overlap_group_ids=groups or [], assignment_reason=reason)

    def _event(self, frame, label, ids, reason, **details):
        self.events.append(dict(frame_index=frame, timestamp_seconds=frame / self.fps,
                                label=label, application_ids=ids, reason=reason,
                                advisory=True, **details))

    def capture(self, frame, observations, links, state, predict=None, tracker_kind='bytetrack'):
        if frame != self.last_frame + 1:
            raise ValueError('Capture every source frame, in order, including empty frames')
        if len(observations) != len(links):
            raise ValueError('Each observation needs its finalized link or None')
        self.last_frame = frame
        identities = state.get('identities', {})
        groups = defaultdict(list)
        for group_id, group in state.get('overlap_groups', {}).items():
            if group.get('state') not in {'EXPIRED', 'RESOLVED'} or (group.get('state') == 'RESOLVED' and group.get('resolved_frame') == frame):
                for app in group.get('identity_ids', []):
                    groups[app].append(int(group_id))
        current, assigned = [], set()
        for item, link in zip(observations, links):
            app = int(link[0]) if link is not None else None
            if app is not None:
                assigned.add(app)
            identity = identities.get(app, {})
            kind = item.get('coordinate_kind', 'observed')
            row = self._row(frame, app, item.get('tracker_id') if tracker_kind == 'bytetrack' else None,
                            item['bbox'], bbox_center(item['bbox']), item.get('confidence'),
                            identity.get('state', identity.get('status', 'CONFIRMED')) if app is not None else 'UNRESOLVED',
                            kind, link[1] if link else 'UNASSIGNED', groups.get(app),
                            reason=link[2] if link else 'No canonical assignment')
            row['underlying_tracker_id'] = item.get('tracker_id')
            row['tracker_kind'] = tracker_kind
            current.append(row)

        # Import actual lifecycle/group events, including termination; no inferred reassignment.
        for key in ('lifecycle_events', 'quality_events'):
            events = state.get(key, [])
            for event in events[self.event_offsets[key]:]:
                copied = json.loads(json.dumps(event))
                app_ids = copied.get('application_ids', [])
                if copied.get('application_id') is not None:
                    app_ids = [copied['application_id']]
                label = copied.get('event', 'TRACKER_EVENT')
                if label == 'ID_REIDENTIFIED':
                    label = 'OCCLUSION_RECOVERY' if bool(copied.get('overlap_partners')) else 'IDENTITY_RECOVERY'
                self.events.append(dict(frame_index=copied.get('frame', frame), label=label,
                                        application_ids=app_ids, source='tracker', details=copied))
            self.event_offsets[key] = len(events)

        for app, identity in identities.items():
            status = identity.get('state', identity.get('status'))
            if app not in assigned and status in {'OCCLUDED', 'LOST', 'UNRESOLVED'} and predict:
                # Use existing tracker's predictor without modifying its history.
                point = predict(identity, frame)
                current.append(self._row(frame, int(app), identity.get('current_byte_id'),
                                         point=point, state=status, kind='predicted',
                                         assignment='UNOBSERVED', groups=groups.get(app),
                                         source='tracker_predicted_center'))
        self._diagnose(frame, current, assigned)
        if not current:
            current.append(self._row(frame, state='NO_OBSERVATIONS', kind='none'))
        self.records.extend(current)
        return current

    def _diagnose(self, frame, current, assigned):
        reversals = []
        for row in current:
            app = row['application_id']
            if app is None or row['coordinate_kind'] not in {'observed', 'tracked'}:
                continue
            previous = self.last.get(app)
            if previous:
                gap = frame - previous['frame_index']
                speed = row['movement_distance'] * self.fps / gap
                if speed > self.config.jump_speed_px_s:
                    self._event(frame, 'POSSIBLE_ID_SWITCH', [app], 'Large observed jump', speed_px_s=speed)
                if (previous['byte_track_id'] is not None and row['byte_track_id'] is not None
                        and previous['byte_track_id'] != row['byte_track_id']
                        and row['assignment_type'] not in {'REASSOCIATED', 'REIDENTIFIED', 'RECOVER', 'OCCLUSION_RECOVERY'}):
                    self._event(frame, 'POSSIBLE_ID_SWITCH', [app], 'ByteTrack association changed; inspect assignment reason')
                vector = np.array([row['x'] - previous['x'], row['y'] - previous['y']]) / gap
                old = self.velocity.get(app)
                if old is not None and np.linalg.norm(old) > 1e-6 and np.linalg.norm(vector) > 1e-6:
                    cosine = float(np.dot(old, vector) / np.linalg.norm(old) / np.linalg.norm(vector))
                    if cosine < -0.7 and row['overlap_group_ids']:
                        reversals.append(row)
                self.velocity[app] = vector
                if gap > 1:
                    self._event(frame, 'IDENTITY_RECOVERY', [app], 'Original canonical ID observed after a gap', gap_frames=gap)
            else:
                for old_id, old in self.last.items():
                    gap = frame - old['frame_index']
                    distance = math.hypot(row['x'] - old['x'], row['y'] - old['y'])
                    if old_id not in assigned and 0 < gap <= self.fps * self.config.nearby_gap_seconds and distance <= self.config.nearby_distance_px:
                        self._event(frame, 'POSSIBLE_FRAGMENTATION', [old_id, app],
                                    'New ID near recently absent ID; proximity is not biological identity',
                                    gap_frames=gap, distance_px=distance)
        for i, first in enumerate(reversals):
            for second in reversals[i + 1:]:
                if set(first['overlap_group_ids']) & set(second['overlap_group_ids']):
                    self._event(frame, 'POSSIBLE_ID_SWITCH', [first['application_id'], second['application_id']],
                                'Both trajectories reversed within a shared overlap group')
        for row in current:
            if row['application_id'] is not None and row['coordinate_kind'] in {'observed', 'tracked'}:
                self.last[row['application_id']] = row

    def write(self, output_dir):
        meta = Path(output_dir) / 'meta'
        meta.mkdir(parents=True, exist_ok=True)
        csv_path = meta / 'grid_tracking_coordinates.csv'
        fields = list(self._row(0)) + ['underlying_tracker_id', 'tracker_kind']
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.records:
                writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        json_path = meta / 'grid_tracking_coordinates.json'
        json_path.write_text(json.dumps(dict(job_id=self.job_id, width=self.width, height=self.height,
                                            fps=self.fps, frame_count=self.last_frame + 1,
                                            coordinate_system='source pixels; origin top-left; x right, y down',
                                            config=asdict(self.config), records=self.records, events=self.events)))
        return {'grid_tracking_csv_path': str(csv_path), 'grid_tracking_json_path': str(json_path)}

    def render(self, source, destination):
        """Render original frames, never the standard annotated video."""
        capture = cv2.VideoCapture(str(source))
        writer = None
        by_frame = defaultdict(list)
        for row in self.records:
            by_frame[row['frame_index']].append(row)
        trails = defaultdict(deque)
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot read {source}')
            writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'MJPG'), self.fps,
                                     (self.width, self.height))
            if not writer.isOpened():
                raise RuntimeError(f'Cannot write {destination}')
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (self.height, self.width):
                    raise ValueError('Source dimensions differ from diagnostic coordinates')
                self.draw(frame, frame_index, by_frame[frame_index], trails)
                writer.write(frame)
                frame_index += 1
            if frame_index != self.last_frame + 1:
                raise ValueError('Source frame count differs from diagnostics')
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    def draw(self, image, frame, records, trails):
        cfg = self.config
        for col in range(1, cfg.columns):
            x = round(col * self.width / cfg.columns)
            cv2.line(image, (x, 0), (x, self.height - 1), (65, 65, 65), 1)
        for row in range(1, cfg.rows):
            y = round(row * self.height / cfg.rows)
            cv2.line(image, (0, y), (self.width - 1, y), (65, 65, 65), 1)
        for row in range(cfg.rows):
            for col in range(cfg.columns):
                cv2.putText(image, f'{column_name(col)}{row + 1}',
                            (int(col * self.width / cfg.columns) + 2, int(row * self.height / cfg.rows) + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .3, (140, 140, 140), 1)
        for app in list(trails):
            while trails[app] and frame - trails[app][0][0] >= cfg.history_length:
                trails[app].popleft()
            if not trails[app]:
                del trails[app]
        occupied_labels = []
        for row in records:
            if not row['in_frame']:
                continue
            app, kind = row['application_id'], row['coordinate_kind']
            x, y = int(round(row['x'])), int(round(row['y']))
            x, y = min(x, self.width - 1), min(y, self.height - 1)
            color = (0, 200, 255) if kind == 'predicted' or app is None else (40, 230, 40)
            if app is not None and kind in {'observed', 'tracked'}:
                trails[app].append((frame, x, y))
                points = list(trails[app])
                for a, b in zip(points, points[1:]):
                    if b[0] == a[0] + 1:  # do not imply observations across missing frames
                        cv2.line(image, a[1:], b[1:], color, 1)
            if kind == 'predicted':
                for angle in range(0, 360, 90):
                    cv2.ellipse(image, (x, y), (6, 6), 0, angle, angle + 45, color, 1)
            else:
                cv2.circle(image, (x, y), 3, color, 1)
            label = f"ID {app if app is not None else '?'} | X:{row['x']:.2f} Y:{row['y']:.2f} | {row['grid_cell']}"
            status = f"{row['identity_state']} {kind.upper()}"
            if row['overlap_group_ids']:
                status += ' Group:' + ','.join(map(str, row['overlap_group_ids']))
            if row['assignment_type'] == 'REIDENTIFIED':
                status += ' RECOVERED'
            label_width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .35, 1)[0][0]
                              for text in (label, status))
            candidates = []
            for dx in (7, -label_width - 7, -label_width // 2):
                for dy in (-8, 28, -40, 60, -72, 92):
                    lx = max(1, min(x + dx, self.width - label_width - 2))
                    ly = max(12, min(y + dy, self.height - 27))
                    box = (lx, ly - 10, lx + label_width, ly + 14)
                    overlap = sum(max(0, min(box[2], b[2]) - max(box[0], b[0])) *
                                  max(0, min(box[3], b[3]) - max(box[1], b[1])) for b in occupied_labels)
                    candidates.append((overlap, math.hypot(lx - x, ly - y), box))
            _, _, box = min(candidates)
            occupied_labels.append(box)
            label_x, label_y = box[0], box[1] + 10
            if math.hypot(label_x - x, label_y - y) > 20:
                cv2.line(image, (x, y), (max(box[0], min(x, box[2])), max(box[1], min(y, box[3]))), color, 1)
            for offset, text in ((0, label), (12, status)):
                cv2.putText(image, text, (label_x, label_y + offset), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 3)
                cv2.putText(image, text, (label_x, label_y + offset), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
        cv2.putText(image, f'Frame {frame} | bbox centers px | dashed=PREDICTED', (4, self.height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)


def coordinate_history(path, application_id):
    data = json.loads(Path(path).read_text())
    rows = [r for r in data['records'] if r['application_id'] == application_id]
    if not rows:
        raise KeyError(application_id)
    observed = [r for r in rows if r['coordinate_kind'] in {'observed', 'tracked'}]
    return dict(job_id=data['job_id'], application_id=application_id,
                first_observed_frame=observed[0]['frame_index'] if observed else None,
                last_observed_frame=observed[-1]['frame_index'] if observed else None,
                coordinate_history=rows,
                grid_cell_history=[{'frame_index': r['frame_index'], 'cell': r['grid_cell'], 'coordinate_kind': r['coordinate_kind']} for r in rows],
                trajectory=[[r['frame_index'], r['x'], r['y']] for r in observed],
                byte_track_history=[{'frame_index': r['frame_index'], 'byte_track_id': r['byte_track_id']} for r in observed],
                events=[e for e in data['events'] if application_id in e.get('application_ids', [])])
