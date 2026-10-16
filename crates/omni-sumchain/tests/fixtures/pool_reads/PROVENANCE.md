# pool_getUnit / pool_getOffer fixtures: real sum-chain server output

Every `*.response.json` here is the verbatim HTTP response body sum-chain's
JSON-RPC server returned for the matching `*.request.json`. Nothing is hand-written.

- **Server:** sum-chain `90c49da13fe4c8848b484f039e4ee12ef29965b9`
  (`rpc: read-only pool_getUnit and pool_getOffer over committed C1 rows (#130)`),
  `RpcServer::start` on 127.0.0.1, answering over plain HTTP/1.1.
- **State:** built with the chain's real model operations (`create_job`, `publish_offer`,
  `reserve_capacity`, `transition_unit`, `put_assignment`, `accept_leaf`) and committed
  through `ComputePoolStore::stage_transition` and the real publish path. The harness
  is copied verbatim from that commit's `crates/rpc/tests/pool_read_rpcs.rs`:
  - job `0x01…`: unit 2 accepted, unit 3 assigned (not accepted), unit 4 blocked on unit 2
    with one required input;
  - offer 3 active with reservation 160 of 1,000; offer 5 inactive with no reservation;
  - offer `0x0b…` active, with every amount at or near `u128::MAX`.

  The `*_gate_closed` cases are a fresh database (the production default: pool gate refused,
  nothing written). `unit_corrupt_row` has one undecodable unit row; the server answers -32603.
- **Generator:** `generator.rs.txt` (sha256 `e0c4fbbe428e7d228ea4a39aead867b60da5074b1c8aa797eda35824721f14e9`). It is not compiled here. To regenerate:
  1. check out sum-chain at the commit above;
  2. copy the generator to `crates/rpc/tests/omninode_pool_fixture_gen.rs`;
  3. run `POOL_FIXTURE_OUT=<dir> cargo test -p sumchain-rpc --locked --test omninode_pool_fixture_gen`.

  The same file serves the seeded state for the live test: `POOL_SERVE_READY=<file> POOL_SERVE_STOP=<file>`.

| file | sha256 |
|---|---|
| `offer_active_reserved.request.json` | `c2647557eabc03c78cd7096953c0f1070c63effc6840176cc31cd420ab98194e` |
| `offer_active_reserved.response.json` | `49aed69b5dc77f3259867af2ae333f7c9bf6e418893b6f3edd0718ece7168a0d` |
| `offer_gate_closed.request.json` | `c2647557eabc03c78cd7096953c0f1070c63effc6840176cc31cd420ab98194e` |
| `offer_gate_closed.response.json` | `1b9f3e593e73354720f2329c7bc53c56abf4ef1c4ce38ecc7d6aa26c211b9685` |
| `offer_inactive.request.json` | `660bff35520c1978238846d3e9fd9df16870cb023decf3c7bb797d8858ba9894` |
| `offer_inactive.response.json` | `56728a647da30f54e1ad03c43f335ec30b04fee5635897264beeb8e38cc3f9fb` |
| `offer_not_held.request.json` | `7820f2b43281a968010111f6bfc0ee67d62ed0636996b82ecefb68b1a766ed6d` |
| `offer_not_held.response.json` | `1b9f3e593e73354720f2329c7bc53c56abf4ef1c4ce38ecc7d6aa26c211b9685` |
| `offer_u128_max.request.json` | `e8312dd09c9f7c98f197390e7c55d34678d5db2d8ff497b462976fe7cc59abf5` |
| `offer_u128_max.response.json` | `fa5c9c7f95e8a930aa7f32bd33656e34406c07ce477efd0755724fd17542811b` |
| `unit_accepted.request.json` | `34aceec1d3c127d79846a13c3bd63138b03a7aeb7e2d8d7ef9070ee2d509f1fe` |
| `unit_accepted.response.json` | `912279411db51c292e58c96afde259df350b60d9e0df456d270733f9224206fe` |
| `unit_assigned.request.json` | `b141f67d2db4dbe831cc8b8d4e99380496d3795aeeb0d17b586d7d8b02cc4bfd` |
| `unit_assigned.response.json` | `67fc90805f2923dce26e58029af5b13cd70066443427e4a0e13408fd51b3c1dd` |
| `unit_blocked_with_input.request.json` | `ac50b26f7b55a1774d445b85d6c4e1cf9811cfdb29a01b590bbce7933687591a` |
| `unit_blocked_with_input.response.json` | `9ab8612af90da5cc4dec94e1161717cc0191984c9f349f99c6ca0758bf513aab` |
| `unit_corrupt_row.request.json` | `b141f67d2db4dbe831cc8b8d4e99380496d3795aeeb0d17b586d7d8b02cc4bfd` |
| `unit_corrupt_row.response.json` | `394b54f6b9b5d05ddd6632cd801e43012dcde658731e36fb5584d529ab8bfba4` |
| `unit_gate_closed.request.json` | `34aceec1d3c127d79846a13c3bd63138b03a7aeb7e2d8d7ef9070ee2d509f1fe` |
| `unit_gate_closed.response.json` | `1b9f3e593e73354720f2329c7bc53c56abf4ef1c4ce38ecc7d6aa26c211b9685` |
| `unit_invalid_params.request.json` | `174f180374284dfb5189f6605bfa77be152e8eab203ef9fda3f060046f0a7ec0` |
| `unit_invalid_params.response.json` | `71483bf7c393401dd2dcda9ec376c83bc158689434b80c0363e4ffdf60e5d1b3` |
| `unit_not_held.request.json` | `0ad9d758e9c3deb14b6cbb29458406472bfbc735d0095df58b3e75d8aff41234` |
| `unit_not_held.response.json` | `1b9f3e593e73354720f2329c7bc53c56abf4ef1c4ce38ecc7d6aa26c211b9685` |
