//! Integrates Newton's 2nd law of motion, applying forces and moving entities according to their velocities.
//!
//! See [`IntegratorPlugin`].

use crate::prelude::*;
use bevy::prelude::*;

/// Integrates Newton's 2nd law of motion, applying forces and moving entities according to their velocities.
///
/// This acts as a prediction for the next positions and orientations of the bodies. The [solver] corrects these predicted
/// positions to follow the rules set by the [constraints].
///
/// The integration scheme used is very closely related to implicit Euler integration.
///
/// The integration systems run in [`SubstepSet::Integrate`].
pub struct IntegratorPlugin;

impl Plugin for IntegratorPlugin {
    fn build(&self, app: &mut App) {
        app.get_schedule_mut(SubstepSchedule)
            .expect("add SubstepSchedule first")
            .add_systems(
                (integrate_pos, integrate_rot)
                    .chain()
                    .in_set(SubstepSet::Integrate),
            );
        app.get_schedule_mut(PhysicsSchedule)
            .expect("add PhysicsSchedule first")
            .add_systems(
                apply_impulses
                    .after(PhysicsStepSet::BroadPhase)
                    .before(PhysicsStepSet::Substeps),
            )
            .add_systems(clear_forces_and_impulses.after(PhysicsStepSet::SpatialQuery));
    }
}

type PosIntegrationComponents = (
    &'static RigidBody,
    &'static Position,
    &'static mut PreviousPosition,
    &'static mut AccumulatedTranslation,
    &'static mut Velocity,
    Option<&'static Damping>,
    Option<&'static GravityScale>,
    &'static ExternalForce,
    &'static Mass,
    &'static InverseMass,
    Option<&'static LockedAxes>,
);

/// Explicitly integrates the positions and linear velocities of bodies taking only external forces
/// like gravity into account. This acts as a prediction for the next positions of the bodies.
fn integrate_pos(
    mut bodies: Query<PosIntegrationComponents, Without<Sleeping>>,
    gravity: Res<Gravity>,
    time: Res<Time>,
) {
    let delta_secs = time.delta_seconds_adjusted();

    for (
        rb,
        pos,
        mut prev_pos,
        mut translation,
        mut vel,
        damping,
        gravity_scale,
        external_force,
        mass,
        inv_mass,
        locked_axes,
    ) in &mut bodies
    {
        prev_pos.0 = pos.0;

        if rb.is_static() {
            continue;
        }

        let locked_axes = locked_axes.map_or(LockedAxes::default(), |locked_axes| *locked_axes);

        // Apply damping, gravity and other external forces
        if rb.is_dynamic() {
            // Apply damping
            if let Some(damping) = damping {
                vel.linear *= 1.0 / (1.0 + delta_secs * damping.linear);
            }

            let effective_mass = locked_axes.apply_to_vec(Vector::splat(mass.0));
            let effective_inv_mass = locked_axes.apply_to_vec(Vector::splat(inv_mass.0));

            // Apply forces
            let gravitation_force =
                effective_mass * gravity.0 * gravity_scale.map_or(1.0, |scale| scale.0);
            let external_forces = gravitation_force + external_force.force();
            let delta_lin_vel = delta_secs * external_forces * effective_inv_mass;
            // avoid triggering bevy's change detection unnecessarily
            if delta_lin_vel != Vector::ZERO {
                vel.linear += delta_lin_vel;
            }
        }
        if vel.linear != Vector::ZERO {
            translation.0 += locked_axes.apply_to_vec(delta_secs * vel.linear);
        }
    }
}

type RotIntegrationComponents = (
    &'static RigidBody,
    &'static mut Rotation,
    &'static mut PreviousRotation,
    &'static mut Velocity,
    Option<&'static Damping>,
    &'static ExternalForce,
    &'static Inertia,
    &'static InverseInertia,
    Option<&'static LockedAxes>,
);

/// Explicitly integrates the rotations and angular velocities of bodies taking only external torque into account.
/// This acts as a prediction for the next rotations of the bodies.
#[cfg(feature = "2d")]
fn integrate_rot(mut bodies: Query<RotIntegrationComponents, Without<Sleeping>>, time: Res<Time>) {
    let delta_secs = time.delta_seconds_adjusted();

    for (
        rb,
        mut rot,
        mut prev_rot,
        mut vel,
        damping,
        external_force,
        _inertia,
        inv_inertia,
        locked_axes,
    ) in &mut bodies
    {
        prev_rot.0 = *rot;

        if rb.is_static() {
            continue;
        }

        let locked_axes = locked_axes.map_or(LockedAxes::default(), |locked_axes| *locked_axes);

        // Apply damping and external torque
        if rb.is_dynamic() {
            // Apply damping
            if let Some(damping) = damping {
                // avoid triggering bevy's change detection unnecessarily
                if vel.angular != 0.0 && damping.angular != 0.0 {
                    vel.angular *= 1.0 / (1.0 + delta_secs * damping.angular);
                }
            }

            let effective_inv_inertia = locked_axes.apply_to_rotation(inv_inertia.0);

            // Apply external torque
            let delta_ang_vel = delta_secs * effective_inv_inertia * external_force.torque();
            // avoid triggering bevy's change detection unnecessarily
            if delta_ang_vel != 0.0 {
                vel.angular += delta_ang_vel;
            }
        }
        // avoid triggering bevy's change detection unnecessarily
        let delta = locked_axes.apply_to_angular_velocity(delta_secs * vel.angular);
        if delta != 0.0 {
            *rot += Rotation::from_radians(delta);
        }
    }
}

/// Explicitly integrates the rotations and angular velocities of bodies taking only external torque into account.
/// This acts as a prediction for the next rotations of the bodies.
#[cfg(feature = "3d")]
fn integrate_rot(mut bodies: Query<RotIntegrationComponents, Without<Sleeping>>, time: Res<Time>) {
    let delta_secs = time.delta_seconds_adjusted();

    for (
        rb,
        mut rot,
        mut prev_rot,
        mut vel,
        damping,
        external_force,
        inertia,
        inv_inertia,
        locked_axes,
    ) in &mut bodies
    {
        prev_rot.0 = *rot;

        if rb.is_static() {
            continue;
        }

        let locked_axes = locked_axes.map_or(LockedAxes::default(), |locked_axes| *locked_axes);

        // Apply damping and external torque
        if rb.is_dynamic() {
            // Apply damping
            if let Some(damping) = damping {
                // avoid triggering bevy's change detection unnecessarily
                if vel.angular != Vector::ZERO && damping.angular != 0.0 {
                    vel.angular *= 1.0 / (1.0 + delta_secs * damping.angular);
                }
            }

            let effective_inertia = locked_axes.apply_to_rotation(inertia.rotated(&rot).0);
            let effective_inv_inertia = locked_axes.apply_to_rotation(inv_inertia.rotated(&rot).0);

            // Apply external torque
            let delta_ang_vel = delta_secs
                * effective_inv_inertia
                * (external_force.torque() - vel.angular.cross(effective_inertia * vel.angular));
            // avoid triggering bevy's change detection unnecessarily
            if delta_ang_vel != Vector::ZERO {
                vel.angular += delta_ang_vel;
            }
        }

        let q = Quaternion::from_vec4(vel.angular.extend(0.0)) * rot.0;
        let effective_dq = locked_axes
            .apply_to_angular_velocity(delta_secs * 0.5 * q.xyz())
            .extend(delta_secs * 0.5 * q.w);
        // avoid triggering bevy's change detection unnecessarily
        let delta = Quaternion::from_vec4(effective_dq);
        if delta != Quaternion::from_xyzw(0.0, 0.0, 0.0, 0.0) {
            rot.0 = (rot.0 + delta).normalize();
        }
    }
}

type ImpulseQueryComponents = (
    &'static RigidBody,
    &'static mut ExternalImpulse,
    &'static mut Velocity,
    &'static Rotation,
    &'static InverseMass,
    &'static InverseInertia,
    Option<&'static LockedAxes>,
);

fn apply_impulses(mut bodies: Query<ImpulseQueryComponents, Without<Sleeping>>) {
    for (rb, impulse, mut vel, rotation, inv_mass, inv_inertia, locked_axes) in &mut bodies {
        if !rb.is_dynamic() {
            continue;
        }

        let locked_axes = locked_axes.map_or(LockedAxes::default(), |locked_axes| *locked_axes);

        let effective_inv_mass = locked_axes.apply_to_vec(Vector::splat(inv_mass.0));
        let effective_inv_inertia = locked_axes.apply_to_rotation(inv_inertia.rotated(rotation).0);

        // avoid triggering bevy's change detection unnecessarily
        let delta_lin_vel = impulse.impulse() * effective_inv_mass;
        let delta_ang_vel = effective_inv_inertia * impulse.angular_impulse();

        if delta_lin_vel != Vector::ZERO {
            vel.linear += delta_lin_vel;
        }
        if delta_ang_vel != AxialVector::ZERO {
            vel.angular += delta_ang_vel;
        }
    }
}

type ForceComponents = (&'static mut ExternalForce, &'static mut ExternalImpulse);
type ForceComponentsChanged = Or<(Changed<ExternalForce>, Changed<ExternalImpulse>)>;

/// Responsible for clearing forces and impulses on bodies.
///
/// Runs in [`PhysicsSchedule`], after [`PhysicsStepSet::SpatialQuery`].
pub fn clear_forces_and_impulses(mut forces: Query<ForceComponents, ForceComponentsChanged>) {
    for (mut force, mut impulse) in &mut forces {
        if !force.persistent {
            force.clear();
        }
        if !impulse.persistent {
            impulse.clear();
        }
    }
}
