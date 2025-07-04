import { z } from "zod";

// Base properties shared across all hero match data.
const baseMatchSchema = z.object({
  kills: z.number().int().min(0, "Kills must be a non-negative integer."),
  deaths: z.number().int().min(0, "Deaths must be a non-negative integer."),
  assists: z.number().int().min(0, "Assists must be a non-negative integer."),
  gold_per_min: z.number().int().min(0),
  hero_damage: z.number().int().min(0),
  turret_damage: z.number().int().min(0),
  damage_taken: z.number().int().min(0),
  teamfight_participation: z.number().int().min(0).max(100),
  positioning_rating: z.enum(["low", "average", "good"]),
  ult_usage: z.enum(["low", "average", "high"]),
  match_duration: z.number().int().min(0),
});

// Schema for heroes without additional specific stats.
const genericHeroSchema = (heroName: string) =>
  baseMatchSchema.extend({
    hero: z.literal(heroName),
  });

// Schema for Franco, who has unique stats.
const francoSchema = baseMatchSchema.extend({
  hero: z.literal("franco"),
  hooks_landed: z.number().int().min(0),
  team_engages: z.number().int().min(0),
  vision_score: z.number().int().min(0),
});

// A discriminated union to enforce hero-specific fields at the type level.
// This ensures that if hero is 'franco', the franco-specific fields are expected.
const matchSchema = z.discriminatedUnion("hero", [
  francoSchema,
  genericHeroSchema("miya"),
  genericHeroSchema("estes"),
  genericHeroSchema("kagura"),
  genericHeroSchema("lancelot"),
  genericHeroSchema("chou"),
  genericHeroSchema("tigreal"),
]);

// A schema to validate the entire `sample_match.json` array.
const matchesSchema = z.array(matchSchema);

// Exporting the Zod schemas and inferred TypeScript types.
export { matchSchema, matchesSchema };
export type Match = z.infer<typeof matchSchema>;
export type Matches = z.infer<typeof matchesSchema>;
