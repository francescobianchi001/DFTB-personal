# Load every converged optimisation movie at once, one visible at a time.
#   vmd -e opt_examp/show_all.tcl
# Then in the Tk console:  m 3      (show molecule 3 alone)
#                          m        (cycle to the next one)
# Bonds come from the final frame for each, via show.tcl.

# `vmd -e` leaves [info script] empty, so find the directory holding show.tcl
set DIR ""
foreach cand [list [file dirname [info script]] opt_examp . \
                   /data/fbianchi/GitHub/DFTB/opt_examp] {
    if {$cand ne "" && [file exists [file join $cand show.tcl]]} {
        set DIR $cand
        break
    }
}
if {$DIR eq ""} { error "show_all.tcl: cannot locate show.tcl" }
source [file join $DIR show.tcl]

set MOVIES {
    ethene_traj.xyz
    butadiene_traj.xyz
    butane_traj.xyz
    benzene_id_traj.xyz
    naphthalene_traj.xyz
    anthracene_traj.xyz
    phenanthrene_traj.xyz
    tetracene_traj.xyz
}

set LOADED {}
foreach f $MOVIES {
    set p [file join $DIR $f]
    if {![file exists $p]} { puts "skip (missing): $f" ; continue }
    lappend LOADED [show $p]
}

proc m {{which -1}} {
    global LOADED CURRENT
    if {![info exists CURRENT]} { set CURRENT 0 }
    if {$which < 0} {
        set CURRENT [expr {($CURRENT + 1) % [llength $LOADED]}]
    } else {
        set CURRENT [expr {$which % [llength $LOADED]}]
    }
    set i 0
    foreach mm $LOADED {
        mol off $mm
        incr i
    }
    set sel [lindex $LOADED $CURRENT]
    mol on $sel
    mol top $sel
    display resetview
    animate goto 0
    puts "showing \[$CURRENT\] [molinfo $sel get name] : [molinfo $sel get numframes] frames"
}

puts "\n==== [llength $LOADED] optimisation movies loaded ===="
set i 0
foreach mm $LOADED {
    puts [format "  \[%d\] %-26s %3d frames" $i [molinfo $mm get name] \
              [molinfo $mm get numframes]]
    incr i
}
puts "  'm' cycles, 'm <n>' picks one; play with the animation controls."
m 0
