from typing import List, Literal, Union, Annotated
from pydantic import BaseModel, Field, conint

# Pydantic models for type-safe data validation, mirroring Zod schemas.


class BaseMatch(BaseModel):
    """Common fields for all matches. Validation rules are defined here."""
    kills: conint(ge=0)
    deaths: conint(ge=0)
    assists: conint(ge=0)
    gold_per_min: conint(ge=0)
    hero_damage: conint(ge=0)
    turret_damage: conint(ge=0)
    damage_taken: conint(ge=0)
    teamfight_participation: conint(ge=0, le=100)
    positioning_rating: Literal['low', 'average', 'good']
    ult_usage: Literal['low', 'average', 'high']
    match_duration: conint(ge=0)


# Schema for Franco, who has unique, required stats.
class FrancoMatch(BaseMatch):
    """Schema for Franco, who has unique, required stats."""
    hero: Literal['franco']
    hooks_landed: conint(ge=0)
    team_engages: conint(ge=0)
    vision_score: conint(ge=0)


# Schemas for other heroes who only require the base stats.
class MiyaMatch(BaseMatch):
    """Schema for Miya."""
    hero: Literal['miya']


class EstesMatch(BaseMatch):
    """Schema for Estes."""
    hero: Literal['estes']


class KaguraMatch(BaseMatch):
    """Schema for Kagura."""
    hero: Literal['kagura']


class LancelotMatch(BaseMatch):
    """Schema for Lancelot."""
    hero: Literal['lancelot']


class ChouMatch(BaseMatch):
    """Schema for Chou."""
    hero: Literal['chou']


class TigrealMatch(BaseMatch):
    """Schema for Tigreal."""
    hero: Literal['tigreal']


class AngelaMatch(BaseMatch):
    """Schema for Angela."""
    hero: Literal['angela']


class FredrinnMatch(BaseMatch):
    """Schema for Fredrinn."""
    hero: Literal['fredrinn']


class HayabusaMatch(BaseMatch):
    """Schema for Hayabusa."""
    hero: Literal['hayabusa']


class UnknownMatch(BaseMatch):
    """Schema for unknown heroes."""
    hero: Literal['unknown']


# A discriminated union to validate against the correct hero schema.
# Pydantic uses the `hero` literal to determine which model to use.
AnyMatch = Annotated[
    Union[
        FrancoMatch,
        MiyaMatch,
        EstesMatch,
        KaguraMatch,
        LancelotMatch,
        ChouMatch,
        TigrealMatch,
        AngelaMatch,
        FredrinnMatch,
        HayabusaMatch,
        UnknownMatch
    ],
    Field(discriminator='hero')
]


# A schema to validate the entire list of matches.
class Matches(BaseModel):
    """A schema to validate the entire list of matches."""
    data: List[AnyMatch] 