from sqlalchemy import Column, Float, Integer, String, Text

from app.database import Base


class Food(Base):
    """
    A single IFCT (or IFCT-style) food entry. One food_name can have multiple
    rows under different region_variant values (e.g. Chicken Biryani —
    Hyderabadi / Awadhi / Kolkata) — this is the regional-variant handling
    layer described in the project architecture.
    """
    __tablename__ = "foods"

    id = Column(Integer, primary_key=True, index=True)
    food_code = Column(String, unique=True, index=True, nullable=False)
    food_name = Column(String, index=True, nullable=False)
    region_variant = Column(String, index=True, default="Standard")
    food_group = Column(String, index=True)
    description = Column(Text)

    calories_kcal = Column(Float, default=0.0)
    protein_g = Column(Float, default=0.0)
    carbs_g = Column(Float, default=0.0)
    fat_g = Column(Float, default=0.0)
    fiber_g = Column(Float, default=0.0)
    calcium_mg = Column(Float, default=0.0)
    iron_mg = Column(Float, default=0.0)

    typical_portion_unit = Column(String, default="katori")
    typical_portion_grams = Column(Float, default=150.0)


class PortionUnit(Base):
    """Maps an Indian serving-unit name (katori, roti, plate, ...) to grams."""
    __tablename__ = "portion_units"

    id = Column(Integer, primary_key=True, index=True)
    unit_name = Column(String, unique=True, index=True, nullable=False)
    grams = Column(Float, nullable=False)
    notes = Column(Text)
