# Load an optimisation movie with correct bonds.
#   vmd -e opt_examp/show.tcl -args opt_examp/cyclohexane_movie.xyz
# or from the Tk console:   source opt_examp/show.tcl ; show benzene
#
# VMD guesses bonds once, from frame 0. In these movies frame 0 is the distorted
# start, where a stretched bond can sit past the covalent cutoff and is then never
# drawn. Taking the topology from the LAST frame (the converged structure) fixes it
# for the whole trajectory.

proc show {name {dir opt_examp}} {
    if {[file exists $name]} {
        set f $name
    } else {
        set f [file join $dir ${name}_movie.xyz]
    }
    set m [mol new $f type xyz waitfor all]

    animate goto end
    mol bondsrecalc $m
    mol reanalyze $m
    animate goto 0

    mol delrep 0 $m
    mol representation CPK 0.28 0.12 24 24
    mol color Element
    mol selection all
    mol addrep $m

    display projection Orthographic
    display depthcue off
    axes location Off
    color Display Background white
    mol showperiodic $m 0 ""
    puts "loaded [file tail $f] : [molinfo $m get numframes] frames, bonds from final frame"
    return $m
}

if {[info exists argv] && [llength $argv] > 0} {
    show [lindex $argv 0]
}
