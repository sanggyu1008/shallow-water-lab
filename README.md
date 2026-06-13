# Shallow Water Lab

Shallow Water Lab은 **천수방정식(shallow water equations)** 을 격자 위에서 유한차분으로 단계적으로 직접 적분하며, 해양/대기 **전진모델(forward model)** 의 핵심 구조를 이해하기 위한 개인 실습 프로젝트입니다.

실제 해양순환모델(ROMS, MOM, HYCOM 등)으로 넘어가기 전에, 회전하는 얕은 유체의 가장 단순한 지배방정식인 천수방정식에서 엇갈린 격자(Arakawa C-grid)·CFL 안정성·지형류 조정·회전계 파동·풍성순환 같은 핵심 개념을 작은 규모에서 먼저 익히는 것을 목표로 합니다.

> 이 프로젝트는 4부작 해양 수치모델 실습 시리즈에서 빠져 있던 **전진모델** 축을 채웁니다.
> - [lorenz-da-lab](https://github.com/sanggyu1008/lorenz-da-lab) — 모델에 관측을 *동화(자료동화)* 합니다.
> - [particle-tracking-lab](https://github.com/sanggyu1008/particle-tracking-lab) — 주어진 속도장을 *따라다닙니다(라그랑지안)*.
> - [advection-diffusion-lab](https://github.com/sanggyu1008/advection-diffusion-lab) — 주어진 속도장 위에서 *트레이서를 수송(오일러리안)* 합니다.
> - **shallow-water-lab** — 흐름 자체를 *격자 PDE로 생성(전진모델)* 합니다. ← 현재 저장소
>
> 그 "모델/속도장"을 격자 위 PDE로 **직접 생성** 하며, Stage 6에서 위 프로젝트들과 연결합니다.

---

## 1. 프로젝트 목표

이 저장소의 주요 목표는 다음과 같습니다.

1. 1차원 선형 천수방정식과 중력파, CFL 조건 이해
2. 엇갈린 격자(staggered / Arakawa C-grid)의 필요성과 구현
3. 2차원 천수방정식의 파동 전파와 경계 처리
4. 코리올리 항과 지형류 조정, 변형 반지름
5. Kelvin파·Rossby파 등 회전계 파동 재현
6. 바람 강제와 마찰로 풍성순환(Stommel·Munk gyre) 재현
7. 모델이 만든 흐름을 입자추적·트레이서 수송·자료동화에 연결

---

## 2. 학습 순서

전체 실습은 다음 순서로 진행합니다.

```text
01. 1차원 천수방정식과 중력파 (staggered grid, CFL)
02. 2차원 천수방정식과 Arakawa C-grid
03. 회전(코리올리)과 지형류 조정 (Rossby 변형 반지름)
04. 회전계 파동: Kelvin파와 Rossby파
05. 풍성순환 (Stommel · Munk gyre, 서안경계강화)
06. double-gyre 순환과 시리즈 결합 (capstone)
```

추천 노트북 구성은 다음과 같습니다.

```text
notebooks/
├── 01_shallow_water_1d.ipynb
├── 02_shallow_water_2d_cgrid.ipynb
├── 03_geostrophic_adjustment.ipynb
├── 04_rotating_waves.ipynb
├── 05_wind_driven_gyre.ipynb
└── 06_double_gyre_and_coupling.ipynb
```

---

## 3. 디렉토리 구조

```text
shallow-water-lab/
├── configs/
│   ├── grid/
│   └── experiment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── environment/
│   ├── environment.yml
│   └── requirements.txt
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── fields/
│   └── logs/
│
├── src/
│   └── shallow_water/
│       ├── grids/      # 엇갈린 격자 (Arakawa C-grid)
│       ├── dynamics/   # 천수방정식 RHS, 코리올리, 강제
│       ├── numerics/   # 시간적분 (leapfrog / forward-backward)
│       ├── forcing/    # 바람 응력, 마찰
│       └── utils/      # 경로/입출력 보조 함수
│
└── tests/
```

---

## 4. 환경 설정

시리즈 4개 프로젝트는 모두 같은 conda 환경 `numlab` 을 공유합니다. 이미 만들었다면 새로 만들 필요 없이 그대로 쓰면 됩니다.

### 4.1 Conda 환경 생성

```bash
conda env create -f environment/environment.yml
conda activate numlab
```

Jupyter Notebook에서 사용할 수 있도록 kernel을 등록합니다.

```bash
python -m ipykernel install --user --name numlab --display-name "Python (numlab)"
```

### 4.2 pip 기반 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
pip install -e .
```

---

## 5. import 방식

이 프로젝트는 `src/` 레이아웃을 사용합니다. editable 설치(`pip install -e .`) 후 어디서나 import 하거나,
노트북 상단의 bootstrap 셀이 `src/`를 `sys.path`에 직접 추가합니다(프로젝트 폴더를 옮겨도 경로가 깨지지 않도록 `find_project_root()` 사용).

```python
from shallow_water.grids.grid1d import StaggeredGrid1D
from shallow_water.grids.cgrid import CGrid
from shallow_water.dynamics.swe2d import ShallowWater2D
from shallow_water.dynamics import diagnostics
from shallow_water.forcing import wind
```

---

## 6. 실습 노트북 (전체 완성)

6단계 노트북이 모두 작성·실행 완료되었습니다. 순서대로 따라가면 됩니다.

```text
notebooks/
├── 01_shallow_water_1d.ipynb          # 1D 중력파·엇갈린 격자·CFL·에너지·leapfrog
├── 02_shallow_water_2d_cgrid.ipynb    # 2D Arakawa C-grid·방사·반사·A/B/C-grid
├── 03_geostrophic_adjustment.ipynb    # 코리올리·지형류 조정·변형반지름·PV 보존
├── 04_rotating_waves.ipynb            # 연안 Kelvin파·β-평면 Rossby파(서향)
├── 05_wind_driven_gyre.ipynb          # Stommel/Munk·서안경계강화·Sverdrup 균형
└── 06_double_gyre_and_coupling.ipynb  # 이중 gyre + 입자추적/트레이서 결합(capstone)
```

구현된 `src` 모듈:

- `grids/` — `StaggeredGrid1D`(1D 엇갈린 격자), `CGrid`(Arakawa C-grid + 차분/보간 연산자)
- `dynamics/` — `swe1d`(FB/leapfrog), `swe2d`(`ShallowWater2D`: 회전·바람·마찰), `diagnostics`
- `numerics/` — `timestep`(적분 드라이버, Robert–Asselin 필터)
- `forcing/` — `wind`(단일/이중 gyre 응력, Stommel/Munk 경계층)

테스트(`tests/`)는 파속·CFL·질량/에너지 보존·지형류 평형·PV·관성진동 등을 검증합니다.

```bash
PYTHONPATH=src python -m pytest         # 34 passed
```

---

## 7. 출력물 관리 규칙

실습 중 생성되는 결과물은 `outputs/` 아래에 저장합니다.

```text
outputs/
├── figures/   # 그림 파일
├── fields/    # 속도장/스냅샷
└── logs/      # 실험 로그
```

큰 출력 파일은 Git에 올리지 않습니다. 단, 디렉토리 구조 유지를 위한 `.gitkeep`은 남깁니다.

---

## 8. 개발 원칙

1. 노트북은 개념 설명과 시각화 중심으로 작성합니다.
2. 반복해서 사용하는 함수는 `src/shallow_water/` 아래로 옮깁니다.
3. 중요한 수치 계산 함수는 `tests/`에서 검증합니다.
4. 실험 설정값은 가능하면 `configs/` 아래에 저장합니다.
5. 출력물은 `outputs/` 아래에 저장합니다.
6. 처음에는 단순하게(선형·1D) 구현하고, 점진적으로(2D·회전·강제) 확장합니다.

---

## 9. 향후 계획

자세한 학습 계획은 다음 문서를 참고합니다.

```text
docs/roadmap.md
```
