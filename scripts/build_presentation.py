#!/usr/bin/env python3
"""Build the four-slide presentation with an embedded video and speaker notes.

Optional authoring dependencies: python-pptx and Pillow. Run from any directory.
PDF export uses LibreOffice separately; the PPTX remains the editable master.
"""
from pathlib import Path
import subprocess
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs'/'presentation'
WORK = ROOT/'work'/'presentation'
REPO = 'https://github.com/WhiteLion1212/MuJoCoTest'
VIDEO = REPO+'/blob/main/docs/blackout_preview.mp4'
PAPER = 'https://arxiv.org/abs/2404.00923'
BG='F6F5F0'; INK='163337'; TEAL='35766F'; MUTED='617171'; ACCENT='C97E51'; LINE='DCE2DC'; WHITE='FFFFFF'

SLIDES = [
    dict(title='Testing SLAM when vision fails', timing='0:00–0:35',
         say='SLAM estimates camera motion while building a map. MM3DGS combines vision, depth, and inertial measurements with a 3D Gaussian map. We built a controlled MuJoCo environment to test what happens when the visual observations become weak or disappear. The aim is a repeatable comparison of tracking methods, especially methods using the IMU.',
         backup='The reference is Sun et al., MM3DGS SLAM, 2024, arXiv:2404.00923. Section III-D in the attached paper uses open-loop inertial propagation to initialize visual tracking and leaves closed-loop fusion with bias estimation as future work. The checked official repository also contains an optional inertial pose-prior loss; it is not a complete bias-estimating inertial factor graph. A blackout motivates testing inertial robustness; it does not establish that tightly coupled fusion is necessary.'),
    dict(title='A reproducible visual–inertial testbed', timing='0:35–1:25',
         say='The scene is a furnished studio made procedurally in Python. Furniture and plants use MuJoCo primitives, and six textures are generated locally. The same 24-second motion is recorded with a 30-hertz RGB-D camera and a 100-hertz IMU. We export calibrated, synchronized measurements, keep ground truth separate for scoring, and replay exactly the same data into each estimator. The three conditions are rich visuals, low texture with blur and dimming, and a complete visual blackout.',
         backup='Scene: studio-v2; no downloaded room, scan, mesh pack, or generated-image service was used. There are six deterministic PNG textures, exported beside scene.xml. Motion: prescribed 6-DoF loop with changing acceleration, yaw, roll, pitch, and height; stationary from 0–2 s and 22–24 s. MuJoCo inverse dynamics supplies the wrench and forward dynamics produces the accelerometer and gyro readings. A 600 Hz sampling grid gives integer strides for 30 Hz cameras and 100 Hz IMU. Both endpoint timestamps are included: 721 images and 2,401 raw IMU samples. A 100 Hz / 30 Hz bridge groups IMU intervals 3/3/4; the initial t=0 sample has no preceding integration interval.\n\nCalibration: T_A_B maps B into A; world z is up; body/IMU axes are x forward, y left, z up; camera axes are x right, y down, z forward. The camera is 0.08 m forward and 0.04 m above the IMU. Accelerometer readings are specific force R_W_B.T @ (a_W - g_W), so a stationary, level sensor reads +9.81 m/s² on body z. Timestamps are integer nanoseconds on a shared simulation clock; IMU is delivered before an image at the same time.\n\nSensor model: nonzero biases and random walks; seed 7 in the shown run. Acceleration white-noise density is 0.002 m/s²/√Hz and gyro density is 0.0002 rad/s/√Hz. Bias random-walk coefficients are 0.0003 (acceleration) and 0.00002 (gyro), with discrete increments scaled by √dt. Initial planted biases are (0.025,-0.015,0.020) m/s² and (0.001,-0.0006,0.0008) rad/s. Sample noise sigma is density × √rate. Replay hides planted bias values, seed, scenario timing, and ground truth from the estimator.\n\nDepth is optical-axis z in metres, zero invalid, with 5 mm independent Gaussian noise and a 0.1–20 m validity range. This is a simple axial-depth model, not stereo reconstruction. Images use instantaneous global exposure; low-texture blur and dimming are image-space approximations. Rolling shutter and clock offsets are not simulated. This is a sensor benchmark rather than a locomotion/controller environment.'),
    dict(title='Demo: four seconds without vision', timing='1:25–2:10, including 24-second video',
         say='The left panel is a clean reference so we can see that the rig keeps moving. It is never given to the estimator. The right panel is the actual camera input. At eight seconds it goes black, and depth is also invalid until twelve seconds. The IMU keeps measuring throughout. When vision returns, we can test whether tracking recovers. This is a video of simulated sensor inputs, not a reconstructed SLAM map.',
         backup='Click the embedded video while presenting slide 3. It is H.264, 960×420, with 721 frames at 30 fps and duration 24.033 seconds including the endpoint frame. Play once without pausing. Point out the beginning of the blackout at 8 s and recovery at 12 s. There are exactly 120 black RGB frames and all corresponding depth pixels are invalid. Outside this interval the scene remains visually rich; the separate low_texture sequence removes material variation and adds blur, dimming, and pixel noise over the same interval.\n\nDepth must also be gated because healthy RGB-D depth can permit tracking despite poor RGB. For other experiments, depth_mode=off provides a monocular sequence and depth_mode=clean provides a depth-assisted control. All matched scenarios share geometry, trajectory, calibration, and IMU noise. The full studio run has byte-identical raw IMU and camera ground truth to the original checkerboard run. Scene appearance changes visual observability, so comparisons should use the same scene version and exported sequence.\n\nPlayback fallback: open ../blackout_preview.mp4 from this folder, or use the online MP4 link on the slide. The embedded video is self-contained in PowerPoint, but video playback varies by presentation application. PDF is a static four-page handout: use its external video link. No CUDA or running simulator is needed to show this clip.'),
    dict(title='Verified simulator. Estimator comparison comes next.', timing='2:10–3:15',
         say='Thirty checks passed for the simulator, including physical IMU output, timing, calibration, depth geometry, and blackout handling. As a sensor sanity check, we initialized both predictors with the true pose and velocity at the start of the outage. Constant velocity ended about 4.44 metres away, while noisy IMU propagation ended about 5 centimetres away. These are not SLAM accuracy results. We still need to run the full MM3DGS pipeline on CUDA and compare visual-only tracking, IMU-initialized tracking, and tightly coupled fusion over multiple seeds and outage lengths.',
         backup='Verification: Linux x86_64, Python 3.12, MuJoCo 3.3.7, EGL/Mesa CPU rendering; pytest reported 30 passed in 167.13 s. Checks include actual accelerometer/gyro versus rigid-body equations; analytic trajectory derivatives; camera extrinsics and depth versus ray intersections; deterministic resets; matched geometry and sensor noise; frame/IMU ordering; ground-truth isolation; metric trajectory scoring; saved scene/texture reload; furniture clearance along the path; and the official loader/result-conversion bridge.\n\nThe sanity check is single-seed inertial dead reckoning, not a visual/inertial estimator. It starts at t=8 s with the oracle (ground-truth) boundary pose and velocity. Biases are estimated from the first 1.5 stationary seconds with known initial orientation; planted bias values are not supplied to propagation. IMU endpoint position error at 12 s is 0.05105 m versus 4.43730 m for constant velocity; outage position RMSE is 0.02199 m versus 2.09482 m. Neither result validates a 3D Gaussian map, reconstruction quality, closed-loop bias estimation, or tight coupling.\n\nA well-initialized loosely coupled filter or inertial propagator can also bridge a short blackout. Tight coupling is most meaningfully tested against such baselines when weak visual constraints remain and during recovery. Even tightly coupled VIO drifts during complete visual loss. Compare: (1) visual tracking plus constant velocity, (2) the paper’s IMU-initialized tracking, (3) joint pose/velocity/bias estimation with visual and inertial residuals. Use identical data, several seeds and outage durations, and the rich and low_texture controls.\n\nReport metric ATE, 1-second translation/rotation RPE, overall and per-interval pose coverage, outage drift, and return-to-vision recovery. The supplied evaluator fixes SE(3) alignment at the first matched pose, does not fit scale, and does not realign after the outage. The official CUDA tracking/mapping pipeline was not run in the CPU-only workspace. Its current unmodified tracker may fail on empty visual/depth masks; this is a benchmark outcome, not a reason to invent measurements.')
]


def rgb(value): return RGBColor.from_string(value)
def rect(slide,x,y,w,h,fill,line=None,rounded=False):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid();shape.fill.fore_color.rgb=rgb(fill)
    if line: shape.line.color.rgb=rgb(line)
    else: shape.line.fill.background()
    if rounded: shape.adjustments[0]=.12
    return shape

def text(slide,x,y,w,h,content,size=20,color=INK,bold=False,align=None):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=box.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0
    for i,line in enumerate(content.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color)
        p.space_after=Pt(5);p.space_before=Pt(0)
        if align is not None:p.alignment=align
    return box

def picture(slide,path,x,y,w,h):
    im=Image.open(path);ratio=im.width/im.height;target=w/h
    pw,ph=(w,w/ratio) if ratio>target else (h*ratio,h)
    return slide.shapes.add_picture(str(path),Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),width=Inches(pw),height=Inches(ph))

def footer(slide,index):
    rect(slide,.55,7.17,12.23,.012,LINE)
    box=text(slide,.55,7.25,10,.17,'MuJoCoTest  /  studio-v2  /  MM3DGS-inspired benchmark',9,color=MUTED)
    box.click_action.hyperlink.address=REPO
    text(slide,12.0,7.23,.78,.23,f'{index} / 4',10,color=MUTED,align=PP_ALIGN.RIGHT)

def new_slide(prs,index,title=None,subtitle=None):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(BG)
    if title:text(s,.55,.42,12.2,.58,title,30,bold=True)
    if subtitle:text(s,.55,1.04,12.2,.4,subtitle,16,color=MUTED)
    footer(s,index)
    return s

def write_notes(slide,data):
    tf=slide.notes_slide.notes_text_frame
    tf.text=f"{data['title']}\nTiming: {data['timing']}\n\nSPEAKING SCRIPT\n{data['say']}\n\nTECHNICAL BACKUP / Q&A\n{data['backup']}\n\nSOURCES\nPaper: {PAPER}\nBenchmark source: {REPO}/tree/3073c7b5fc443c7adaf081013325ef04fd883b06\nImplementation notes: {REPO}/blob/main/README.md\nVerification: {REPO}/blob/main/VALIDATION.md"


def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    poster=WORK/'video_poster.png'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','4','-i',str(ROOT/'docs/blackout_preview.mp4'),'-frames:v','1',str(poster)],check=True)
    p=Presentation();p.slide_width=Inches(13.333333);p.slide_height=Inches(7.5)
    p.core_properties.title='Testing SLAM when vision fails'
    p.core_properties.subject='Four-slide MuJoCo benchmark presentation with embedded video'
    p.core_properties.author='MuJoCoTest'

    # 1: The problem, and why this benchmark exists.
    s=new_slide(p,1)
    text(s,.6,.48,5.3,.3,'MM3DGS SLAM  /  MUJOCO TESTBED',13,color=TEAL,bold=True)
    text(s,.6,1.22,5.35,2.5,'Testing SLAM\nwhen vision\nfails',42,bold=True)
    text(s,.6,3.98,5.1,1.0,'A reproducible benchmark for\nvisual–inertial tracking.',22,color=MUTED)
    ref=text(s,.6,5.38,5.1,.65,'Inspired by MM3DGS SLAM\nSun et al., 2024',15,color=TEAL)
    ref.click_action.hyperlink.address=PAPER
    picture(s,ROOT/'docs/studio.png',6.25,1.20,6.53,4.90)
    text(s,6.25,6.17,6.53,.25,'Original procedural studio · no downloaded scene assets',12,color=MUTED,align=PP_ALIGN.CENTER)
    rect(s,.55,6.62,12.23,.43,TEAL,rounded=True)
    text(s,.75,6.70,11.85,.30,'Goal: compare tracking through weak or missing visual observations.',18,color=WHITE)

    # 2: Repeatable measurements, without overwhelming the audience with settings.
    s=new_slide(p,2,'A reproducible visual–inertial testbed','Procedural studio · synchronized sensors · ground truth reserved for scoring')
    stats=[('24 s','smooth 6-DoF sequence'),('30 Hz','RGB + metric depth'),('100 Hz','noisy, biased IMU')]
    for i,(value,label) in enumerate(stats):
        x=.55+i*4.16
        rect(s,x,1.70,3.91,1.22,WHITE,LINE,True)
        text(s,x+.20,1.83,3.5,.56,value,34,color=TEAL,bold=True)
        text(s,x+.20,2.45,3.5,.30,label,17,color=MUTED)
    labels=[('1  MuJoCo rig','Prescribed motion\nKnown camera–IMU calibration'),('2  Sensor streams','RGB, depth and raw IMU\nReplay the same data'),('3  Estimator + scoring','Official-code data bridge\nATE, RPE and pose coverage')]
    for i,(title,body) in enumerate(labels):
        x=.55+i*4.16
        rect(s,x,3.46,3.91,1.52,INK,rounded=True)
        text(s,x+.20,3.65,3.52,.35,title,21,color=WHITE,bold=True)
        text(s,x+.20,4.13,3.52,.66,body,17,color='D9E8E4')
        if i<2:text(s,x+3.91,3.94,.25,.4,'›',22,color=TEAL,align=PP_ALIGN.CENTER)
    text(s,.55,5.40,12,.30,'THREE MATCHED VISUAL CONDITIONS',13,color=TEAL,bold=True)
    for i,(label,desc) in enumerate([('Rich','normal visual reference'),('Low texture','dimming + blur'),('Blackout','RGB and depth unavailable')]):
        x=.55+i*4.16
        rect(s,x,5.90,.07,.68,ACCENT)
        text(s,x+.18,5.88,3.6,.35,label,21,bold=True)
        text(s,x+.18,6.30,3.6,.35,desc,16,color=MUTED)

    # 3: A self-contained embedded video, plus an online fallback link.
    s=new_slide(p,3,'Demo: four seconds without vision','Left: clean reference only. Right: actual camera input. Click the video to play.')
    button=rect(s,10.70,.49,2.08,.36,TEAL,rounded=True)
    button.click_action.hyperlink.address=VIDEO
    link=text(s,10.78,.56,1.91,.20,'Open MP4 fallback',11,color=WHITE,align=PP_ALIGN.CENTER)
    link.click_action.hyperlink.address=VIDEO
    s.shapes.add_movie(str(ROOT/'docs/blackout_preview.mp4'),Inches(.55),Inches(1.50),Inches(12.23),Inches(5.350625),poster_frame_image=str(poster),mime_type='video/mp4')
    for x,w,color in [(.55,12.23*8/24,TEAL),(.55+12.23*8/24,12.23*4/24,ACCENT),(.55+12.23*12/24,12.23*12/24,TEAL)]:
        rect(s,x,6.91,w,.07,color)
    for x,w,label in [(.55,4.07,'0–8 s: normal observations'),(4.63,2.04,'8–12 s: blackout'),(6.67,6.11,'12–24 s: visual recovery')]:
        text(s,x,7.00,w,.19,label,10,color=MUTED,align=PP_ALIGN.CENTER)

    # 4: Evidence, limits, and the next experiment.
    s=new_slide(p,4,'Verified simulator. Estimator comparison comes next.','The current evidence validates the testbed and the sensor streams.')
    rect(s,.55,1.70,4.48,3.34,WHITE,LINE,True)
    text(s,.80,1.93,3.95,.30,'VERIFIED',13,color=TEAL,bold=True)
    text(s,.80,2.40,3.95,.68,'30 checks passed',30,color=INK,bold=True)
    text(s,.80,3.30,3.95,1.35,'Physical IMU output and calibration\nCamera/depth geometry and timing\nAll 120 blackout frames invalid',17,color=MUTED)
    rect(s,5.28,1.70,7.50,3.34,INK,rounded=True)
    text(s,5.55,1.93,6.95,.30,'SANITY CHECK  /  4-SECOND OUTAGE',13,color='C6E3D9',bold=True)
    text(s,5.55,2.55,4.5,.38,'Predictor',17,color='D9E8E4')
    text(s,9.90,2.55,2.55,.38,'Endpoint error',17,color='D9E8E4',align=PP_ALIGN.RIGHT)
    rect(s,5.55,3.03,6.95,.012,'4B6467')
    text(s,5.55,3.23,4.5,.40,'Constant velocity',22,color=WHITE)
    text(s,9.90,3.18,2.55,.48,'4.44 m',28,color='E7AC86',bold=True,align=PP_ALIGN.RIGHT)
    text(s,5.55,3.86,4.5,.40,'Noisy IMU propagation',22,color=WHITE)
    text(s,9.90,3.81,2.55,.48,'0.051 m',28,color='A4D8C6',bold=True,align=PP_ALIGN.RIGHT)
    text(s,5.55,4.52,6.95,.32,'Oracle starting pose + velocity · one seed · sensor check, not SLAM accuracy',12,color='D9E8E4')
    text(s,.55,5.43,12.23,.66,'Next: visual-only → IMU initialization → tightly coupled fusion',23,color=TEAL,bold=True)
    text(s,.55,6.18,12.23,.72,'Run full MM3DGS on CUDA; compare error, coverage and recovery across seeds.\nThe full SLAM pipeline has not run here; necessity of tight coupling remains unproven.',16,color=MUTED)

    for slide,data in zip(p.slides,SLIDES): write_notes(slide,data)
    path=OUT/'MuJoCo_SLAM_4_slides.pptx';p.save(path)
    # LibreOffice omits video poster frames in PDF export. Use an explicitly
    # static copy for the handout while preserving the movie in the main PPTX.
    handout=Presentation(path)
    video_slide=handout.slides[2]
    for shape in list(video_slide.shapes):
        if shape.shape_type == MSO_SHAPE_TYPE.MEDIA:
            x,y,w,h=shape.left,shape.top,shape.width,shape.height
            video_slide.shapes._spTree.remove(shape._element)
            video_slide.shapes.add_picture(str(poster),x,y,width=w,height=h)
    for timing in list(video_slide._element.findall(qn('p:timing'))):
        video_slide._element.remove(timing)
    handout.save(WORK/'MuJoCo_SLAM_handout_source.pptx')
    notes=['# Presenter notes — four-slide MuJoCo SLAM presentation','',
           'Planned length: about 3–4 minutes, including the 24-second video. The PPTX contains exactly four slides and the same speaker notes. The PDF is a static four-page handout.','',
           '## Before presenting','',
           '- Open `MuJoCo_SLAM_4_slides.pptx` in PowerPoint and test the embedded video on slide 3. Enter Slide Show and click the video; it is not configured to autoplay.','- The MP4 is embedded, so PowerPoint does not need an internet connection or CUDA. Compatibility with other presentation applications varies. Keep `../blackout_preview.mp4` available as a fallback, or use the online video link.','- For the PDF, open the MP4 separately. No simulator, Python environment or GPU is required during the talk.','- State that the video shows inputs, and that the numerical example is an oracle-initialized sensor check. Do not present it as a completed MM3DGS or tight-coupling experiment.','']
    for i,d in enumerate(SLIDES,1):
        notes += [f"## Slide {i} — {d['title']}",'',f"Timing: {d['timing']}",'','### Suggested script','',d['say'],'','### Technical backup / likely questions','',d['backup'],'']
    notes += ['## Reproduce or extend the demonstration','',
              'The scene snapshot used to prepare this deck is commit `3073c7b5fc443c7adaf081013325ef04fd883b06` (`studio-v2`). Use the same scene and measurements across estimators.','',
              '```bash','python -m venv .venv','source .venv/bin/activate','pip install -e ".[test]"','mm3dgs-bench generate --scenario suite --out runs/presentation --width 480 --height 360','python scripts/make_preview.py runs/presentation/blackout --out runs/presentation_preview.mp4','pytest -q','```','',
              'For the official CUDA pipeline, follow the upstream MM3DGS environment instructions and this repository’s README. The simulator environment and official CUDA environment are separate. The supplied runner can prepare configs with `--prepare-only`; keep `tracking.use_gt_pose=false` and compare CV versus IMU initialization. Add a separate tightly coupled estimator through the documented replay interface.','',
              '## Sources and attribution','',f'- Paper: Lisong C. Sun et al., *MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using Vision, Depth, and Inertial Measurements* (2024), {PAPER}.',f'- Official project: https://vita-group.github.io/MM3DGS-SLAM/; code: https://github.com/VITA-Group/MM3DGS-SLAM. The data bridge was checked against upstream commit `1fae9f5d1ad50a6159717b711ae54325422202e2`.',f'- Simulator and implementation details: {REPO}/blob/3073c7b5fc443c7adaf081013325ef04fd883b06/README.md.',f'- Recorded validation: {REPO}/blob/3073c7b5fc443c7adaf081013325ef04fd883b06/VALIDATION.md.',f'- Video: {VIDEO}. All studio geometry and six textures are generated by the repository; there is no external graphics pack.','',
              '## Rebuild the slides','',
              '`python scripts/build_presentation.py` requires optional authoring packages `python-pptx` and `Pillow`, plus `ffmpeg` for the poster frame. To regenerate the PDF with LibreOffice, export `work/presentation/MuJoCo_SLAM_handout_source.pptx` (the static poster copy created by the builder), then save the four-page PDF as `docs/presentation/MuJoCo_SLAM_4_slides.pdf`. The editable master contains the embedded movie.','']
    (OUT/'PRESENTER_NOTES.md').write_text('\n'.join(notes))
    print(path)
    print('Slides:',len(p.slides),'— video embedded on slide 3; speaker notes on every slide')


if __name__=='__main__': main()
