# Test Fixtures

This directory houses the minimal reproducible test-bed used by the unit and
integration suites. Populated progressively as sprints land.

Required structure (to be created by Sprint 3-5 fixture work):

```
fixtures/
  sample_repo/
    package.json
    src/
      app/api/commissions/[id]/duplicate/route.ts
      lib/services/commission.service.ts
      lib/repositories/commission.repository.ts
      lib/types/listing.ts
      components/dashboard/CommissionListingCard.tsx
      components/dashboard/DialogManager.tsx
      hooks/useCommissions.ts
    docs/
      srs.md
      sdd.md
  sample_gt.json
```

The fixture must exercise every one of the 13 edge types and every one of
the 9 node types. See 12_project_structure_and_sprints.md §4 for details.
