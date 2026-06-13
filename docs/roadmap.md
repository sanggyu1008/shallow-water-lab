# Shallow Water Lab 학습 로드맵

이 문서는 Shallow Water Lab 프로젝트의 전체 학습 순서를 정리한 문서입니다.

목표는 **천수방정식(shallow water equations, SWE)** 을 격자 위에서 직접 적분하며, 해양/대기 전진모델(forward model)의 핵심 요소 — 엇갈린 격자(Arakawa C-grid), CFL 안정성, 지형류 조정, 회전계 파동(Kelvin·Rossby), 풍성순환 — 을 단계적으로 구현하고 이해하는 것입니다.

선형화한 1.5층(또는 단일층) 천수방정식은 다음과 같습니다.

$$
\frac{\partial u}{\partial t} - f v = -g\frac{\partial \eta}{\partial x}, \qquad
\frac{\partial v}{\partial t} + f u = -g\frac{\partial \eta}{\partial y}, \qquad
\frac{\partial \eta}{\partial t} + H\!\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) = 0
$$

여기서 $(u,v)$ 는 유속, $\eta$ 는 자유표면 변위, $H$ 는 평균 수심, $f$ 는 코리올리 파라미터, $g$ 는 중력가속도입니다.

> 이 프로젝트는 시리즈에서 빠져 있던 **전진모델(forward model)** 축을 채웁니다.
> `lorenz-da-lab` 은 모델에 관측을 *동화* 하고, `particle-tracking-lab` 은 주어진 속도장을 *따라다닙니다*.
> 이 프로젝트는 그 "모델/속도장"을 격자 위 PDE로 **직접 생성** 합니다.
> Stage 6에서 이 모델이 만든 흐름을 두 프로젝트에 연결합니다.

---

## Stage 1. 1차원 천수방정식과 중력파

노트북:

```text
notebooks/01_shallow_water_1d.ipynb
```

목표:

1. 선형 1차원 천수방정식 $\partial_t u = -g\,\partial_x \eta,\ \partial_t \eta = -H\,\partial_x u$ 이해
2. **엇갈린 격자(staggered grid)**: $u$ 와 $\eta$ 를 반 칸 어긋나게 배치하는 이유
3. forward-backward(또는 leapfrog) 시간적분 구현
4. 중력파 속도 $c=\sqrt{gH}$ 와 CFL 조건 $c\,\Delta t/\Delta x \le 1$ 확인
5. 정상파(standing wave)·진행파, 댐 붕괴(dam-break)류 초기조건 실험
6. 에너지(운동+위치) 보존 점검

핵심 개념:

- 천수방정식의 유도(정수압·얕은 물 근사)
- staggered grid 와 격자 잡음(grid-scale noise) 억제
- 중력파, CFL
- 시간적분 스킴(leapfrog, forward-backward)

---

## Stage 2. 2차원 천수방정식과 Arakawa C-grid

노트북:

```text
notebooks/02_shallow_water_2d_cgrid.ipynb
```

목표:

1. 2차원 선형 천수방정식을 **Arakawa C-grid** 에 구현
2. $u, v, \eta$ 의 격자 배치와 공간 차분
3. 중력파의 2차원 전파와 경계 반사
4. 2차원 CFL 조건
5. 격자 종류(A/B/C-grid)의 차이 개념 소개

핵심 개념:

- Arakawa A/B/C-grid 와 분산관계(dispersion) 정확도
- 2차원 파동 전파
- 경계조건(벽, 방사)

---

## Stage 3. 회전(코리올리)과 지형류 조정

노트북:

```text
notebooks/03_geostrophic_adjustment.ipynb
```

목표:

1. 코리올리 항 추가 (f-plane)
2. 고전적 **지형류 조정(geostrophic adjustment)** 문제: 초기 단차가 어떻게 조정되는가
3. 변형 반지름(Rossby radius of deformation) $L_d = \sqrt{gH}/f$ 의 의미
4. 관성-중력파(inertia-gravity wave)와 최종 지형류 평형 상태
5. 위치소용돌이도(potential vorticity) 보존 확인

핵심 개념:

- 지형류 평형(geostrophic balance)
- 변형 반지름과 조정 규모
- 관성진동, 위치소용돌이도 보존

---

## Stage 4. 회전계 파동: Kelvin파와 Rossby파

노트북:

```text
notebooks/04_rotating_waves.ipynb
```

목표:

1. 경계에 붙어 전파하는 **연안 Kelvin파(coastal Kelvin wave)** 재현
2. $\beta$-평면에서 **Rossby파** 재현 (서향 위상전파)
3. 분산관계와 위상/군속도 관찰
4. 파동이 해양 순환에 갖는 의미 토의

핵심 개념:

- Kelvin파(경계 포획, 비분산)
- Rossby파($\beta$ 효과, 서향 전파)
- 분산관계

---

## Stage 5. 풍성순환 (Stommel · Munk gyre)

노트북:

```text
notebooks/05_wind_driven_gyre.ipynb
```

목표:

1. 바람 응력(wind stress) 강제와 마찰(바닥/측면) 추가
2. **Stommel**(바닥 마찰)과 **Munk**(측면 마찰) 순환해 재현
3. **서안경계강화(western boundary intensification)** 관찰
4. Sverdrup 균형과 내부 해양 흐름
5. 정상상태 도달까지의 spin-up

핵심 개념:

- Sverdrup 균형, 서안경계류
- 바닥/측면 마찰의 역할
- 바람 강제 순환의 spin-up

---

## Stage 6. double-gyre 순환과 시리즈 결합 (capstone)

노트북:

```text
notebooks/06_double_gyre_and_coupling.ipynb
```

목표:

1. 풍성순환 설정으로 **double-gyre 형태의 흐름** 생성
2. 모델이 만든 속도장을 NetCDF/배열로 저장
3. `particle-tracking-lab` 으로 그 흐름에서 **입자추적**
4. `advection-diffusion-lab` 으로 그 흐름 위에서 **트레이서 수송**
5. (선택) `lorenz-da-lab` 의 EnKF로 이 모델 상태를 **동화**
6. 세 프로젝트가 하나의 모델 위에서 만나는 통합 실험

핵심 개념:

- 전진모델 → 입자추적 / 트레이서 / 자료동화로의 연결
- 해석적 속도장(double-gyre) vs 모델이 생성한 속도장
- 수치모델 워크플로 전체 조망

---

## 개발 원칙

이 프로젝트에서는 다음 원칙을 따른다.

1. 노트북은 설명과 시각화 중심으로 작성한다.
2. 반복해서 사용하는 함수는 `src/shallow_water/`로 옮긴다.
3. 실험 설정값은 `configs/` 아래에 저장한다.
4. 출력 결과는 `outputs/` 아래에 저장한다.
5. 중요한 수치 계산 함수는 `tests/`에서 검증한다.
6. 처음에는 단순하게(선형·1D) 구현하고, 이후 점진적으로(2D·회전·강제) 확장한다.

---

## 진행 상태

```text
스캐폴딩(구조·환경·README·roadmap)            (완료)
Stage 1  01_shallow_water_1d.ipynb            (완료)
Stage 2  02_shallow_water_2d_cgrid.ipynb      (완료)
Stage 3  03_geostrophic_adjustment.ipynb      (완료)
Stage 4  04_rotating_waves.ipynb              (완료)
Stage 5  05_wind_driven_gyre.ipynb            (완료)
Stage 6  06_double_gyre_and_coupling.ipynb    (완료)
src 모듈(grids/dynamics/numerics/forcing) + tests 34개  (완료)
```

**시리즈 완료.** 검증된 핵심 결과:

- 1D 봉우리가 ±c 로 분리(파속 오차 ~0.1%), FB 에너지 변동 1% 미만, 엇갈린 leapfrog 은 CFL≤0.5.
- 2D 닫힌 분지에서 질량 기계정밀도 보존, 고리 반지름 r=ct.
- 지형류 조정: 폭 ~Ld 의 jet, PV 시간평균 보존(~10%), 지형류 평형.
- Kelvin파: 비분산 c=√(gH), offshore 감쇠규모 = Ld(700 km); Rossby파: 서향 위상속도 측정 -3.94 vs 이론 -3.90 m/s.
- 풍성순환: 서안경계강화, 내부 Sverdrup 균형(상관 1.00), Stommel δ_S=r/β vs Munk δ_M=(A_h/β)^{1/3}.
- 이중 gyre 속도장 → 입자추적(궤적=유선) + 트레이서 수송 결합.

확장 후보: 비선형 항(이류) 추가로 불안정 jet·중규모 eddy, 1.5층/2층 모델, NetCDF 입출력,
EnKF 로 천수모델 상태 동화.

---

## 참고 문헌

- Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics, 2nd ed.* Cambridge University Press. — 천수방정식·지형류 조정·파동의 표준 교재.
- Cushman-Roisin, B., Beckers, J.-M. (2011). *Introduction to Geophysical Fluid Dynamics, 2nd ed.* Academic Press. — 수치 구현과 격자 관점.
- Arakawa, A., Lamb, V. R. (1977). *Computational design of the basic dynamical processes of the UCLA general circulation model.* Methods in Computational Physics, 17, 173–265. — A/B/C-grid.
- Stommel, H. (1948). *The westward intensification of wind-driven ocean currents.* Trans. AGU, 29, 202–206.
- Munk, W. H. (1950). *On the wind-driven ocean circulation.* J. Meteorology, 7, 79–93.
