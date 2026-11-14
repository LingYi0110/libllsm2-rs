# Attribution & Third-Party Notices

This project is a safe Rust rewrite of **libllsm2 v2.1**, licensed under **GPL-3.0-or-later** (see `LICENSE`).

- **libllsm2 v2.1**
  Copyright (c) 2017–2019 Kanru Hua, GPL-3.0-or-later.
  Core speech analysis, Layer 0/1 modeling, Pulse-by-Pulse synthesis, Coder, and streaming synthesis architecture are adapted from libllsm2 v2.1.

- **ciglet**
  Selected DSP algorithms and Chebyshev filter tables are adapted from ciglet (BSD 3-Clause):

  ```text
  Copyright (c) 2016-2019, Kanru Hua
  All rights reserved.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  ```

- **Test Audio Fixtures** (`crates/llsm2/tests/fixtures/audio/`)
  - `arctic_a0001.wav` — CMU ARCTIC speech database (`http://festvox.org/cmu_arctic/`)
  - `are-you-ready.wav` — unfa (`https://freesound.org/people/unfa/sounds/258342/`)
